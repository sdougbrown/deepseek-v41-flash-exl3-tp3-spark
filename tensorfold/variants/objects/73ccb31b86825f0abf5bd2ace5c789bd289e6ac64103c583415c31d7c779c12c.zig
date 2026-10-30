//! A stepped-prefill backend for LaneHost's resumable tests: the one-piece-per-call contract over a
//! `lanes.fake.Fake`. Each call appends real prompt history and restores cached state before the first yield.
//! One `prefill_step` call is one piece; a decode round is never forced from inside the backend.
const std = @import("std");
const Allocator = std.mem.Allocator;
const lanes = @import("lanes");
const be = lanes.backend;
const Stream = lanes.Stream;

/// What ran, in order: a piece (with the core's round count when it ended), a prefilled event, a delivered token,
/// a finished event. Pieces and tokens share one list so tests assert the interleaving.
pub const Trace = union(enum) {
    piece: struct { at: usize, steps: u64 },
    first: u64,
    token: struct { id: u64, token: u32 },
    finished: struct { id: u64, reason: []const u8 },
};

pub const Stepped = struct {
    gpa: Allocator,
    target: *lanes.fake.Fake,
    /// Maximum pieces per prompt pass; each nonempty piece appends real fake-model history.
    pieces: usize = 3,
    trace: ?*std.ArrayList(Trace) = null,
    /// The core's round counter, read when a piece ends (tests pass `&core.steps`).
    steps: ?*const u64 = null,
    /// A first piece refuses with error.ContextFull while another stream's pass completed (the pool is busy).
    refuse_concurrent: bool = false,
    releases: usize = 0,
    cancel_next: bool = false,
    inner: be.Backend = undefined,
    at: std.AutoHashMapUnmanaged(*Stream, u32) = .empty, // pieces completed per stream
    done: std.ArrayList(*Stream) = .empty, // streams whose pass completed, until their release

    pub fn deinit(x: *Stepped) void {
        x.at.deinit(x.gpa);
        x.done.deinit(x.gpa);
    }

    pub fn backend(x: *Stepped) be.Backend {
        return .{ .ptr = x, .vtable = &.{
            .prefill = prefill,
            .prefill_step = prefillStep,
            .first = first,
            .queue = queue,
            .read = read,
            .verify = verify,
            .keep = keep,
            .draft = draft,
            .release = release,
        } };
    }

    fn self(ptr: *anyopaque) *Stepped {
        return @ptrCast(@alignCast(ptr));
    }

    fn note(x: *Stepped, t: Trace) void {
        if (x.trace) |list| list.append(x.gpa, t) catch {};
    }

    fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
        return self(ptr).inner.prefill(s);
    }

    fn prefillStep(ptr: *anyopaque, s: *Stream) anyerror!bool {
        const x = self(ptr);
        const at = x.at.get(s) orelse 0;
        if (x.refuse_concurrent and at == 0 and x.done.items.len > 0) return error.ContextFull;
        if (x.cancel_next or s.isCancelled()) {
            x.cancel_next = false;
            return error.Cancelled;
        }
        x.note(.{ .piece = .{ .at = at + 1, .steps = if (x.steps) |p| p.* else 0 } });
        const done = try x.target.prefillPart(s, x.pieces);
        try x.at.put(x.gpa, s, @intCast(at + 1));
        if (done) try x.done.append(x.gpa, s);
        return done;
    }

    fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        return self(ptr).inner.first(s, position);
    }

    fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        return self(ptr).inner.queue(s, feed, position);
    }

    fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return self(ptr).inner.read(handle);
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        return self(ptr).inner.verify(windows, out);
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        return self(ptr).inner.keep(windows, paths);
    }

    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        return self(ptr).inner.draft(requests);
    }

    fn release(ptr: *anyopaque, s: *Stream) void {
        const x = self(ptr);
        x.releases += 1;
        _ = x.at.remove(s);
        for (x.done.items, 0..) |d, i| if (d == s) {
            _ = x.done.orderedRemove(i);
            break;
        };
        x.inner.release(s);
    }
};
