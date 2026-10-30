const std = @import("std");
const abi = @import("cuda/abi.zig");
const Driver = @import("cuda/driver.zig").Driver;
const cuda = struct {
    pub const graph = @import("cuda/graph.zig");
    pub const Stream = @import("cuda/stream.zig").Stream;
};
pub const std_options: std.Options = .{ .logFn = silent };
fn silent(comptime level: std.log.Level, comptime scope: @TypeOf(.default), comptime format: []const u8, args: anytype) void {
    _ = level; _ = scope; _ = format; _ = args;
}
const prompt = struct { const Shared = struct {}; };
const Engine = struct { s: cuda.Stream };
const Round = struct { kk: usize = 0, has_cand: bool = false, has_cblk: bool = false };
const Chunk = struct {};
const Caches = struct {};
const Pool = struct {};
const Drafter = struct {
    fn body(_: *Drafter, _: *const Engine, _: *const Chunk, _: *const Pool, _: usize, _: usize, _: ?*anyopaque) !void {}
};
fn stretch(_: *const Engine, _: *Round, _: *const Chunk, _: *const Caches, _: []const u64, _: *Carry, _: usize, _: usize, _: ?*anyopaque) !void {}
const State = struct {
    alive: [64]bool = @splat(false),
    next: usize = 1,
    destroy_calls: usize = 0,
    double_destroys: usize = 0,
    stale_launches: usize = 0,
    launches: usize = 0,
    fail_launches: usize = 0,
    fail_instantiate: bool = false,
};
var state: State = .{};
fn begin(_: abi.Stream, _: abi.CaptureMode) callconv(.c) abi.Result { return 0; }
fn mockEndCapture(_: abi.Stream, graph: *abi.Graph) callconv(.c) abi.Result { graph.* = @ptrFromInt(1); return 0; }
fn destroyGraph(_: abi.Graph) callconv(.c) abi.Result { return 0; }
fn instantiate(exec: *abi.GraphExec, _: abi.Graph, _: u64) callconv(.c) abi.Result {
    if (state.fail_instantiate) return 1;
    const id = state.next;
    state.next += 1;
    state.alive[id] = true;
    exec.* = @ptrFromInt(id);
    return 0;
}
fn launch(exec: abi.GraphExec, _: abi.Stream) callconv(.c) abi.Result {
    const id = @intFromPtr(exec.?);
    state.launches += 1;
    if (!state.alive[id]) { state.stale_launches += 1; return 1; }
    if (state.fail_launches != 0) { state.fail_launches -= 1; return 1; }
    return 0;
}
fn destroyExec(exec: abi.GraphExec) callconv(.c) abi.Result {
    const id = @intFromPtr(exec.?);
    state.destroy_calls += 1;
    if (!state.alive[id]) state.double_destroys += 1;
    state.alive[id] = false;
    return 0;
}
fn errorText(_: abi.Result, text: *?[*:0]const u8) callconv(.c) abi.Result { text.* = "injected mock error"; return 0; }
fn driver() Driver {
    var d: Driver = undefined;
    d.api.cuStreamBeginCapture_v2 = begin;
    d.api.cuStreamEndCapture = mockEndCapture;
    d.api.cuGraphDestroy = destroyGraph;
    d.api.cuGraphInstantiateWithFlags = instantiate;
    d.api.cuGraphLaunch = launch;
    d.api.cuGraphExecDestroy = destroyExec;
    d.api.cuGetErrorName = errorText;
    d.api.cuGetErrorString = errorText;
    return d;
}
fn make(comptime T: type, gpa: std.mem.Allocator, cap: usize) T {
    if (@typeInfo(@TypeOf(T.init)).@"fn".param_types.len == 2) return T.init(gpa, cap);
    return T.init(gpa);
}
fn callRoundAt(g: *Graphs, e: *const Engine, stretch_index: u32) !void {
    var rd: Round = .{};
    var cy: Carry = .{ .h = 1, .spare = 2, .pre = 3, .pre_f = 4 };
    return g.run(.{ .rows = 1, .bucket = 1024, .stretch = stretch_index }, e, &rd, &.{}, &.{}, &.{}, &cy, 0, 0);
}
fn callRound(g: *Graphs, e: *const Engine) !void {
    return callRoundAt(g, e, 0);
}
fn callPass(g: *PassGraphs, e: *const Engine) !void {
    var dr: Drafter = .{};
    return g.run(&dr, e, &.{}, &.{}, 1, 1);
}
fn call(comptime T: type, g: *T, e: *const Engine) !void {
    if (T == Graphs) return callRound(g, e);
    return callPass(g, e);
}
