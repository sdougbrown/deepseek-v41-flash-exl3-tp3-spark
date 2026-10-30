fn retryAfterFailure(comptime T: type) !void {
    state = .{ .fail_launches = 1 };
    const d = driver();
    const e: Engine = .{ .s = .{ .d = &d, .handle = @ptrFromInt(1) } };
    var g = T.init(probe_allocator);
    defer g.deinit();
    try std.testing.expectError(error.CudaFailed, call(T, &g, &e));
    try call(T, &g, &e);
    try std.testing.expectEqual(@as(usize, 0), state.stale_launches);
}
fn teardownAfterFailure(comptime T: type) !void {
    state = .{ .fail_launches = 1 };
    const d = driver();
    const e: Engine = .{ .s = .{ .d = &d, .handle = @ptrFromInt(1) } };
    var g = T.init(probe_allocator);
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
    var g = T.init(failing.allocator());
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
    var g = T.init(probe_allocator);
    defer g.deinit();
    try std.testing.expectError(error.CudaFailed, call(T, &g, &e));
    try std.testing.expectEqual(@as(usize, 0), g.count());
    try std.testing.expectEqual(@as(usize, 0), state.destroy_calls);
}

fn case0() !void { try retryAfterFailure(Graphs); }
fn case1() !void { try retryAfterFailure(PassGraphs); }
fn case2() !void { try teardownAfterFailure(Graphs); }
fn case3() !void { try teardownAfterFailure(PassGraphs); }
fn case4() !void { try insertionFailure(Graphs); }
fn case5() !void { try insertionFailure(PassGraphs); }
fn case6() !void { try instantiateFailure(Graphs); }
fn case7() !void { try instantiateFailure(PassGraphs); }

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
    };
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
