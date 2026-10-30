fn retryAfterFailure(comptime T: type) !void {
    state = .{ .fail_launches = 1 };
    const d = driver();
    const e: Engine = .{ .s = .{ .d = &d, .handle = @ptrFromInt(1) } };
    var g = make(T, probe_allocator, 0);
    defer g.deinit();
    try std.testing.expectError(error.CudaFailed, call(T, &g, &e));
    try call(T, &g, &e);
    try std.testing.expectEqual(@as(usize, 0), state.stale_launches);
}
fn teardownAfterFailure(comptime T: type) !void {
    state = .{ .fail_launches = 1 };
    const d = driver();
    const e: Engine = .{ .s = .{ .d = &d, .handle = @ptrFromInt(1) } };
    var g = make(T, probe_allocator, 0);
    try std.testing.expectError(error.CudaFailed, call(T, &g, &e));
    g.deinit();
    try std.testing.expectEqual(@as(usize, 1), state.destroy_calls);
    try std.testing.expectEqual(@as(usize, 0), state.double_destroys);
}
fn insertionFailure(comptime T: type) !void {
    state = .{};
    var failing = std.testing.FailingAllocator.init(probe_allocator, .{ .fail_index = 0 });
    const d = driver();
    const e: Engine = .{ .s = .{ .d = &d, .handle = @ptrFromInt(1) } };
    var g = make(T, failing.allocator(), 0);
    defer g.deinit();
    try std.testing.expectError(error.OutOfMemory, call(T, &g, &e));
    try std.testing.expectEqual(@as(usize, 0), g.count());
    try std.testing.expectEqual(@as(usize, 1), state.destroy_calls);
    try std.testing.expectEqual(@as(usize, 0), state.double_destroys);
}
fn instantiateFailure(comptime T: type) !void {
    state = .{ .fail_instantiate = true };
    const d = driver();
    const e: Engine = .{ .s = .{ .d = &d, .handle = @ptrFromInt(1) } };
    var g = make(T, probe_allocator, 0);
    defer g.deinit();
    try std.testing.expectError(error.CudaFailed, call(T, &g, &e));
    try std.testing.expectEqual(@as(usize, 0), g.count());
    try std.testing.expectEqual(@as(usize, 0), state.destroy_calls);
}

fn evictionDestroysOnce() !void {
    state = .{};
    const d = driver();
    const e: Engine = .{ .s = .{ .d = &d, .handle = @ptrFromInt(1) } };
    var g = make(Graphs, probe_allocator, 1);
    try callRoundAt(&g, &e, 0);
    try callRoundAt(&g, &e, 1); // evicts stretch 0
    try std.testing.expectEqual(@as(usize, 1), g.count());
    try std.testing.expectEqual(@as(usize, 1), state.destroy_calls);
    try callRoundAt(&g, &e, 0); // captured again, evicts stretch 1
    try callRoundAt(&g, &e, 0); // replayed
    g.deinit();
    try std.testing.expectEqual(@as(usize, 3), state.destroy_calls);
    try std.testing.expectEqual(@as(usize, 0), state.double_destroys);
    try std.testing.expectEqual(@as(usize, 0), state.stale_launches);
    try std.testing.expectEqual(@as(usize, 4), state.launches);
}
fn evictionThenFailedLaunch() !void {
    state = .{};
    const d = driver();
    const e: Engine = .{ .s = .{ .d = &d, .handle = @ptrFromInt(1) } };
    var g = make(Graphs, probe_allocator, 1);
    try callRoundAt(&g, &e, 0);
    state.fail_launches = 1;
    try std.testing.expectError(error.CudaFailed, callRoundAt(&g, &e, 1));
    try callRoundAt(&g, &e, 1);
    g.deinit();
    try std.testing.expectEqual(@as(usize, 2), state.destroy_calls);
    try std.testing.expectEqual(@as(usize, 0), state.double_destroys);
    try std.testing.expectEqual(@as(usize, 0), state.stale_launches);
}

fn case0() !void { try retryAfterFailure(Graphs); }
fn case1() !void { try retryAfterFailure(PassGraphs); }
fn case2() !void { try teardownAfterFailure(Graphs); }
fn case3() !void { try teardownAfterFailure(PassGraphs); }
fn case4() !void { try insertionFailure(Graphs); }
fn case5() !void { try insertionFailure(PassGraphs); }
fn case6() !void { try instantiateFailure(Graphs); }
fn case7() !void { try instantiateFailure(PassGraphs); }
fn case8() !void { try evictionDestroysOnce(); }
fn case9() !void { try evictionThenFailedLaunch(); }

var probe_allocator: std.mem.Allocator = undefined;
pub fn main() !void {
    var heap: std.heap.DebugAllocator(.{}) = .init;
    defer { if (heap.deinit() != .ok) @panic("probe leaked memory"); }
    probe_allocator = heap.allocator();
    const Case = struct { name: []const u8, run: *const fn () anyerror!void };
    const cases = [_]Case{
        .{ .name = "round: a retry never launches a destroyed graph", .run = case0 },
        .{ .name = "drafter: a retry never launches a destroyed graph", .run = case1 },
        .{ .name = "round: failed first launch tears down exactly once", .run = case2 },
        .{ .name = "drafter: failed first launch tears down exactly once", .run = case3 },
        .{ .name = "round: failed cache allocation still frees the unowned executable", .run = case4 },
        .{ .name = "drafter: failed cache allocation still frees the unowned executable", .run = case5 },
        .{ .name = "round: failed instantiate creates no cached executable", .run = case6 },
        .{ .name = "drafter: failed instantiate creates no cached executable", .run = case7 },
    } ++ (if (capped) [_]Case{
        .{ .name = "round: an evicted graph is destroyed once and never launched again", .run = case8 },
        .{ .name = "round: a failed first launch after an eviction leaves a valid graph to retry", .run = case9 },
    } else [_]Case{});
    var failures: usize = 0;
    for (cases) |case| {
        case.run() catch |err| {
            std.debug.print("FAIL {s}: {s}\n", .{ case.name, @errorName(err) });
            failures += 1;
            continue;
        };
        std.debug.print("PASS {s}\n", .{case.name});
    }
    std.debug.print("PROBE {d} passed, {d} failed\n", .{ cases.len - failures, failures });
    if (failures != 0) return error.ProbeFailed;
}
