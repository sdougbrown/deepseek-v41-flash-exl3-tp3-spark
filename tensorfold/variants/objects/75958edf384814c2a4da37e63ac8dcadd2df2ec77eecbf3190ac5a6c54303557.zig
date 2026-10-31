//! The real host loop over incremental fake-model prompt work, without threads or GPU access.
const std = @import("std");
const api = @import("../engine_api.zig");
const lanes = @import("lanes");
const stepped = @import("stepped.zig");
const LaneHost = @import("../lane_host.zig").LaneHost;
const pc = api.prompt_cache;
const a = std.testing.allocator;
const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };

const Fixture = struct {
    cfg: lanes.Config,
    fake: lanes.fake.Fake,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine,
    host: LaneHost,
    step: stepped.Stepped = undefined,
    trace: std.ArrayList(stepped.Trace) = .empty,
    done: std.AutoHashMapUnmanaged(api.Id, api.Reason) = .empty,
    stopped: bool = false,
    is_stepped: bool,

    fn init(f: *Fixture, parts: ?usize) !void {
        f.* = .{
            .cfg = try lanes.Config.init(a, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7),
            .fake = .{ .gpa = a }, .core = undefined, .host = undefined,
            .is_stepped = parts != null,
        };
        f.core = lanes.Engine.init(a, &f.cfg, f.fake.backend(), f.clock.clock());
        f.host = LaneHost.init(a, std.testing.io, &f.core, .{ .lanes = 8 });
        if (parts) |n| {
            f.step = .{ .gpa = a, .target = &f.fake, .pieces = n, .trace = &f.trace, .steps = &f.core.steps };
            f.step.inner = f.fake.backend();
            f.core.backend = f.step.backend();
        }
    }

    fn stop(f: *Fixture) void {
        if (!f.stopped) {
            f.host.stop();
            f.stopped = true;
        }
    }
    fn deinit(f: *Fixture) void {
        f.stop();
        if (f.is_stepped) f.step.deinit();
        f.core.deinit();
        f.fake.deinit();
        f.cfg.deinit(a);
        f.trace.deinit(a);
        f.done.deinit(a);
    }
    fn event(ctx: *anyopaque, id: api.Id, ev: *const api.Event) void {
        const f: *Fixture = @ptrCast(@alignCast(ctx));
        switch (ev.*) {
            .prefilled => f.trace.append(a, .{ .first = id }) catch {},
            .tokens => |values| for (values) |token| { f.trace.append(a, .{ .token = .{ .id = id, .token = token } }) catch {}; },
            .finished => |end| {
                f.trace.append(a, .{ .finished = .{ .id = id, .reason = @tagName(end.reason) } }) catch {};
                f.done.put(a, id, end.reason) catch {};
            },
        }
    }
    fn submit(f: *Fixture, id: api.Id, request: *const api.Request) !void {
        try f.host.engine().submit(id, request, .{ .ctx = f, .event = event });
    }
    fn ready(f: *Fixture, id: api.Id) bool {
        for (f.trace.items) |entry| if (entry == .first and entry.first == id) return true;
        return false;
    }
    fn until(f: *Fixture, id: api.Id, first: bool) !void {
        var count: usize = 0;
        while (if (first) !f.ready(id) else !f.done.contains(id)) {
            try std.testing.expect(count < 10000);
            try std.testing.expect(f.host.iterate());
            count += 1;
        }
    }
    fn tokens(f: *Fixture, id: api.Id) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(a);
        for (f.trace.items) |entry| if (entry == .token and entry.token.id == id) try out.append(a, entry.token.token);
        return out.toOwnedSlice(a);
    }
};

fn incumbent(f: *Fixture) !void {
    const static_request = struct { const value: api.Request = .{ .prompt = &prompt, .max_tokens = 10000 }; };
    try f.submit(1, &static_request.value);
    try f.until(1, true);
}

test "incumbent tokens appear between prefill pieces and newcomer tokens wait for completion" {
    var f: Fixture = undefined;
    try f.init(3);
    defer f.deinit();
    try incumbent(&f);
    const mark = f.trace.items.len;
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 4 };
    try f.submit(2, &request);
    try std.testing.expect(f.host.iterate());
    try std.testing.expectEqual(@as(usize, 1), f.host.pending.items.len);
    try std.testing.expectEqual(@as(usize, 1), f.core.activeCount());
    try std.testing.expect(!f.ready(2));
    try f.until(2, false);
    var last_piece_step: ?u64 = null;
    var pieces: usize = 0;
    var last_piece: usize = mark;
    var incumbent_tokens: usize = 0;
    for (f.trace.items[mark..], mark..) |entry, index| switch (entry) {
        .piece => |part| {
            if (last_piece_step) |last| try std.testing.expect(part.steps > last);
            last_piece_step = part.steps;
            last_piece = index;
            pieces += 1;
        },
        .token => |token| {
            if (token.id == 1 and pieces > 0 and pieces < 3) incumbent_tokens += 1;
            if (token.id == 2) try std.testing.expect(pieces == 3 and index > last_piece);
        },
        else => {},
    };
    try std.testing.expectEqual(@as(usize, 3), pieces);
    try std.testing.expect(incumbent_tokens > 0);
    try std.testing.expect(!f.done.contains(1));
}

test "completing a pending prefill cannot start another piece before decode" {
    var f: Fixture = undefined;
    try f.init(3);
    defer f.deinit();
    try incumbent(&f);
    const mark = f.trace.items.len;
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 4 };
    try f.submit(2, &request);
    try f.submit(3, &request);
    try f.until(2, false);
    try f.until(3, false);
    var last: ?u64 = null;
    var count: usize = 0;
    for (f.trace.items[mark..]) |entry| if (entry == .piece) {
        if (last) |previous| try std.testing.expect(entry.piece.steps > previous);
        last = entry.piece.steps;
        count += 1;
    };
    try std.testing.expectEqual(@as(usize, 6), count);
}

test "decode debt also gates the first piece of the next admitted request" {
    var f: Fixture = undefined;
    try f.init(1);
    defer f.deinit();
    try incumbent(&f);
    f.host.debt.owed_ms = 1000;
    f.host.debt.rounds = 1;
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 4 };
    try f.submit(2, &request);
    try std.testing.expect(f.host.iterate());
    try std.testing.expect(!f.ready(2));
    try std.testing.expectEqual(@as(usize, 1), f.host.queued.items.len);
    try std.testing.expect(f.host.debt.owed_ms > 0);
}

test "completion preserves prefill debt until incumbents repay it before the next admission" {
    var f: Fixture = undefined;
    try f.init(3);
    defer f.deinit();
    try incumbent(&f);
    f.host.pending_limit = 1; // one pass at a time: request 3 is the admission after request 2 completes
    const LongPiece = struct {
        fn finish(ctx: *anyopaque, _: *lanes.Stream, part: usize) void {
            const debt: *@import("../lane_host.zig").Debt = @ptrCast(@alignCast(ctx));
            if (part == 2) debt.owed_ms = 1000000;
        }
    };
    f.fake.prefill_hook = LongPiece.finish;
    f.fake.prefill_hook_ctx = &f.host.debt;
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 1 };
    try f.submit(2, &request);
    try f.submit(3, &request);
    try f.until(2, false);
    const mark = f.trace.items.len;
    try std.testing.expect(f.host.iterate());
    try std.testing.expect(!f.ready(3));
    try std.testing.expectEqual(@as(usize, 1), f.host.queued.items.len);
    for (f.trace.items[mark..]) |entry| try std.testing.expect(entry != .piece);
}

test "queued cancellation and backend cancellation release partial prefill exactly once" {
    for ([_]bool{ false, true }) |from_backend| {
        var f: Fixture = undefined;
        try f.init(5);
        defer f.deinit();
        try incumbent(&f);
        const request: api.Request = .{ .prompt = &prompt, .max_tokens = 4 };
        try f.submit(2, &request);
        try std.testing.expect(f.host.iterate());
        const released = f.step.releases;
        if (from_backend) f.step.cancel_next = true else f.host.engine().cancel(2);
        try f.until(2, false);
        try std.testing.expectEqual(api.Reason.cancelled, f.done.get(2).?);
        try std.testing.expectEqual(released + 1, f.step.releases);
        try std.testing.expectEqual(@as(usize, 0), f.host.pending.items.len);
        try std.testing.expectEqual(@as(usize, 1), f.fake.lanes.count());
        const tokens = try f.tokens(2);
        defer a.free(tokens);
        try std.testing.expectEqual(@as(usize, 0), tokens.len);
    }
}

test "stop releases pending and queued requests before their owners disappear" {
    var f: Fixture = undefined;
    try f.init(5);
    defer f.deinit();
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 4 };
    try f.submit(1, &request);
    try f.submit(2, &request);
    try std.testing.expect(f.host.iterate());
    try std.testing.expectEqual(@as(usize, 1), f.host.pending.items.len);
    f.stop();
    try std.testing.expectEqual(api.Reason.cancelled, f.done.get(1).?);
    try std.testing.expectEqual(api.Reason.cancelled, f.done.get(2).?);
    try std.testing.expectEqual(@as(usize, 1), f.step.releases);
}

test "legacy blocking scheduling does not create pending prefills" {
    var f: Fixture = undefined;
    try f.init(null);
    defer f.deinit();
    f.fake.prefill_chunks = 3;
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 4 };
    try f.submit(1, &request);
    try f.submit(2, &request);
    try f.until(1, false);
    try f.until(2, false);
    try std.testing.expectEqual(@as(usize, 6), f.fake.prefill_count);
    try std.testing.expectEqual(@as(usize, 0), f.host.pending.items.len);
}

test "ContextFull retries at queue front only after an incumbent releases its lane" {
    var f: Fixture = undefined;
    try f.init(2);
    defer f.deinit();
    f.step.refuse_concurrent = true;
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 8 };
    try f.submit(1, &request);
    try f.until(1, true);
    try f.submit(2, &request);
    try std.testing.expect(f.host.iterate());
    try std.testing.expectEqual(@as(usize, 1), f.host.queued.items.len);
    try std.testing.expect(!f.ready(2));
    try f.until(2, false);
    try std.testing.expectEqual(api.Reason.length, f.done.get(2).?);
}

test "foreground FIFO remains ahead of background work" {
    var f: Fixture = undefined;
    try f.init(2);
    defer f.deinit();
    const plain: api.Request = .{ .prompt = &prompt, .max_tokens = 4 };
    var background = plain;
    background.background = true;
    try f.submit(1, &background);
    try f.submit(2, &plain);
    try f.submit(3, &plain);
    for ([_]api.Id{ 1, 2, 3 }) |id| try f.until(id, false);
    var order: [3]api.Id = undefined;
    var n: usize = 0;
    for (f.trace.items) |entry| if (entry == .first) {
        order[n] = entry.first;
        n += 1;
    };
    try std.testing.expectEqualSlices(api.Id, &.{ 2, 3, 1 }, order[0..n]);
}

test "solo pieces drain without forced decode between them" {
    var f: Fixture = undefined;
    try f.init(5);
    defer f.deinit();
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 4 };
    try f.submit(1, &request);
    try f.until(1, false);
    var count: usize = 0;
    for (f.trace.items) |entry| if (entry == .piece) {
        try std.testing.expectEqual(@as(u64, 0), entry.piece.steps);
        count += 1;
    };
    try std.testing.expectEqual(@as(usize, 5), count);
}

test "restored entry can be evicted after the first piece without dangling references" {
    var f: Fixture = undefined;
    try f.init(3);
    defer f.deinit();
    const Snap = struct {
        fn bytes(_: *anyopaque, _: u32) u64 { return 1; }
        fn save(_: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
            const list = try a.create(std.ArrayList(u32));
            list.* = .empty;
            try list.appendSlice(a, prompt[0..at]);
            return list;
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {}
        fn drop(_: *anyopaque, saved: pc.Saved) void {
            const list: *std.ArrayList(u32) = @ptrCast(@alignCast(saved));
            list.deinit(a);
            a.destroy(list);
        }
    };
    var context: u8 = 0;
    var store = pc.Store.init(a, .{ .ptr = &context, .vtable = &.{ .bytes = Snap.bytes, .save = Snap.save, .restore = Snap.restore, .drop = Snap.drop } }, .{ .min_prompt = 0 }, 100);
    defer store.deinit();
    try std.testing.expect(store.keep(prompt[0..4], 4, null, &.{}));
    f.host.cache = &store;
    defer { f.host.cache = null; }
    const request: api.Request = .{ .prompt = &prompt, .max_tokens = 4 };
    try f.submit(1, &request);
    try std.testing.expect(f.host.iterate());
    try std.testing.expectEqual(@as(usize, 1), f.host.pending.items.len);
    try std.testing.expectEqual(@as(u64, 1), store.counts.hits);
    try store.clear();
    try f.until(1, false);
    f.host.cache = null;
    try std.testing.expectEqual(api.Reason.length, f.done.get(1).?);
}

test "fairness repays completed decode time without banking idle credit" {
    var debt: @import("../lane_host.zig").Debt = .{};
    debt.piece(12, 0.5);
    try std.testing.expect(debt.pieceDue(0));
    try std.testing.expect(!debt.pieceDue(1));
    debt.decoded(3);
    try std.testing.expectEqual(@as(f64, 9), debt.owed_ms);
    try std.testing.expect(!debt.pieceDue(1));
    debt.decoded(9);
    try std.testing.expect(debt.pieceDue(1));
    debt.decoded(1000);
    debt.piece(7, 0.5);
    try std.testing.expectEqual(@as(f64, 7), debt.owed_ms);
    debt.decoded(-10);
    try std.testing.expectEqual(@as(f64, 7), debt.owed_ms);
}

test "eight queued mixed sampled and greedy streams preserve blocking outputs and capacity" {
    var blocking: Fixture = undefined;
    try blocking.init(null);
    defer blocking.deinit();
    var resumable: Fixture = undefined;
    try resumable.init(3);
    defer resumable.deinit();
    var requests: [8]api.Request = undefined;
    for (&requests, 0..) |*request, i| {
        request.* = .{
            .prompt = prompt[0 .. 4 + i % 5], .max_tokens = @intCast(12 + i % 4),
            .sampling = if (i % 2 == 0) null else .{ .seed = @intCast(i) },
        };
        try blocking.submit(i + 1, request);
        try resumable.submit(i + 1, request);
    }
    for ([_]*Fixture{ &blocking, &resumable }) |f| {
        var iterations: usize = 0;
        while (f.done.count() < 8) {
            try std.testing.expect(iterations < 10000);
            try std.testing.expect(f.host.iterate());
            try std.testing.expect(f.host.admitted.items.len <= 8);
            try std.testing.expect(f.core.activeCount() + f.host.pending.items.len <= 8);
            iterations += 1;
        }
    }
    for (1..9) |id| {
        const expected = try blocking.tokens(id);
        defer a.free(expected);
        const actual = try resumable.tokens(id);
        defer a.free(actual);
        try std.testing.expectEqualSlices(u32, expected, actual);
    }
}

/// A prompt sixteen pieces long at `Stepped.pieces = 16` (four positions a piece), where `prompt[0..4]` takes four
/// pieces of one position.
const long_prompt: [64]u32 = blk: {
    var ids: [64]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast(i % 7 + 3);
    break :blk ids;
};

fn firstIndex(f: *Fixture, id: api.Id) ?usize {
    for (f.trace.items, 0..) |entry, i| if (entry == .first and entry.first == id) return i;
    return null;
}

fn firstToken(f: *Fixture, id: api.Id) ?usize {
    for (f.trace.items, 0..) |entry, i| if (entry == .token and entry.token.id == id) return i;
    return null;
}

/// The reply is the fake model's greedy continuation of `ids`, `n` tokens long.
fn expectReply(f: *Fixture, id: api.Id, ids: []const u32, n: usize) !void {
    const tokens = try f.tokens(id);
    defer a.free(tokens);
    try std.testing.expectEqual(n, tokens.len);
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(a);
    try history.appendSlice(a, ids);
    for (tokens) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(a, t);
    }
}

test "a short prefill queued behind a long pending one decodes before the long one completes" {
    var f: Fixture = undefined;
    try f.init(16);
    defer f.deinit();
    const long: api.Request = .{ .prompt = &long_prompt, .max_tokens = 4 };
    const short: api.Request = .{ .prompt = prompt[0..4], .max_tokens = 4 };
    try f.submit(1, &long);
    for (0..3) |_| try std.testing.expect(f.host.iterate());
    try std.testing.expect(!f.ready(1));
    try f.submit(2, &short);
    try f.until(1, false);
    try f.until(2, false);
    try std.testing.expect(firstToken(&f, 2).? < firstIndex(&f, 1).?);
    try std.testing.expectEqual(api.Reason.length, f.done.get(1).?);
    try std.testing.expectEqual(api.Reason.length, f.done.get(2).?);
    try expectReply(&f, 1, &long_prompt, 4);
    try expectReply(&f, 2, prompt[0..4], 4);
}

test "cancelling either of two pending prefills leaves the other's reply intact" {
    for ([_]api.Id{ 1, 2 }) |gone| {
        var f: Fixture = undefined;
        try f.init(16);
        defer f.deinit();
        const long: api.Request = .{ .prompt = &long_prompt, .max_tokens = 4 };
        const short: api.Request = .{ .prompt = prompt[0..4], .max_tokens = 4 };
        try f.submit(1, &long);
        for (0..3) |_| try std.testing.expect(f.host.iterate());
        try f.submit(2, &short);
        try std.testing.expect(f.host.iterate());
        try std.testing.expectEqual(@as(usize, 2), f.host.pending.items.len);
        f.host.engine().cancel(gone);
        const other: api.Id = 3 - gone;
        try f.until(gone, false);
        try f.until(other, false);
        try std.testing.expectEqual(api.Reason.cancelled, f.done.get(gone).?);
        try std.testing.expectEqual(api.Reason.length, f.done.get(other).?);
        const ids: []const u32 = if (other == 1) &long_prompt else prompt[0..4];
        try expectReply(&f, other, ids, 4);
        const dropped = try f.tokens(gone);
        defer a.free(dropped);
        try std.testing.expectEqual(@as(usize, 0), dropped.len);
        try std.testing.expectEqual(@as(usize, 2), f.step.releases); // each stream released exactly once
        try std.testing.expectEqual(@as(usize, 0), f.fake.lanes.count());
        try std.testing.expectEqual(@as(usize, 0), f.host.pending.items.len);
    }
}

test "a steady supply of short prefills cannot hold a long one back" {
    // one position: each short pass ends in its admission's piece; four: short passes stay pending beside the long one
    for ([_]usize{ 1, 4 }) |size| {
        var f: Fixture = undefined;
        try f.init(16);
        defer f.deinit();
        const long: api.Request = .{ .prompt = &long_prompt, .max_tokens = 1 };
        const short: api.Request = .{ .prompt = prompt[0..size], .max_tokens = 1 };
        try f.submit(1, &long);
        var id: api.Id = 2;
        var count: usize = 0;
        while (!f.done.contains(1)) : (count += 1) {
            try std.testing.expect(count < 2000);
            if (f.host.queued.items.len == 0) {
                try f.submit(id, &short);
                id += 1;
            }
            try std.testing.expect(f.host.iterate());
        }
        try std.testing.expectEqual(api.Reason.length, f.done.get(1).?);
    }
}

test "a prompt with images waits for the pending prefills, then prefills alone" {
    var f: Fixture = undefined;
    try f.init(16);
    defer f.deinit();
    const long: api.Request = .{ .prompt = &long_prompt, .max_tokens = 1 };
    const pictured: api.Request = .{ .prompt = prompt[0..4], .max_tokens = 1, .images = &.{.{ .at = 0, .tokens = 1, .bytes = "" }} };
    const short: api.Request = .{ .prompt = prompt[0..4], .max_tokens = 1 };
    try f.submit(1, &long);
    try std.testing.expect(f.host.iterate());
    try f.submit(2, &pictured);
    try std.testing.expect(f.host.iterate());
    try f.submit(3, &short);
    for ([_]api.Id{ 1, 2, 3 }) |id| try f.until(id, false);
    var order: [3]api.Id = undefined;
    var n: usize = 0;
    for (f.trace.items) |entry| if (entry == .first) {
        order[n] = entry.first;
        n += 1;
    };
    try std.testing.expectEqualSlices(api.Id, &.{ 1, 2, 3 }, order[0..n]);
}
