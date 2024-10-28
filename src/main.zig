const std = @import("std");

fn despaceScalar(inp: []const u8, outp: []u8) []u8 {
    var out_i: usize = 0;
    for (inp) |ch| {
        outp[out_i] = ch;
        if (ch != ' ' and ch != '\n' and ch != '\t') out_i += 1;
    }
    return outp[0..out_i];
}

fn despaceStdlib(inp: []const u8, outp: []u8) []u8 {
    var out_i: usize = 0;

    // var iter = std.mem.tokenizeAny(u8, inp, " \n\t");
    // while (iter.next()) |buf| {
    //     @memcpy(outp[out_i..][0..buf.len],  buf);
    //     out_i += buf.len;
    // }
    // ^even slower than scalar

    var inp_i: usize = 0;
    while (inp_i < inp.len) {
        var tok_len = inp.len - inp_i;
        for (" \t\n") |delim| {
            if (std.mem.indexOfScalar(u8, inp[inp_i..][0..tok_len], delim)) |new_tok_len| {
                tok_len = new_tok_len;
            }
        }
        @memcpy(outp[out_i..][0..tok_len], inp[inp_i..][0..tok_len]);
        out_i += tok_len;
        inp_i += tok_len + 1;
    }

    return outp[0..out_i];
}

// size_t despace_avx512(const char *src, char *dst, size_t len)
extern fn despace_avx512(src: [*]const u8, dst: [*]u8, len: usize) callconv(.C) usize;

fn despaceAvx512(inp: []const u8, outp: []u8) []u8 {
    return outp[0..despace_avx512(inp.ptr, outp.ptr, inp.len)];
}

pub fn main() !void {
    const buffer_size = 1 << 20;
    const processed_data = 1 << 30;
    const iterations = processed_data / buffer_size;
    const bytes: []u8 = try std.heap.page_allocator.alloc(u8, buffer_size);
    defer std.heap.page_allocator.free(bytes);
    var rng = std.Random.DefaultPrng.init(0);
    const random = rng.random();
    for (bytes) |*e| {
        e.* = 1 + random.int(u8) % 127; // valid ascii chars
    }
    var timer = try std.time.Timer.start();
    const output_buf: []u8 = try std.heap.page_allocator.alloc(u8, buffer_size);
    defer std.heap.page_allocator.free(output_buf);
    _ = despaceScalar(bytes, output_buf);
    _ = timer.lap();
    for (0..iterations) |_| {
        std.mem.doNotOptimizeAway(despaceScalar(bytes, output_buf));
        std.mem.doNotOptimizeAway(bytes);
        std.mem.doNotOptimizeAway(output_buf);
    }
    const scalar_time = timer.lap();
    _ = despaceStdlib(bytes, output_buf);
    _ = timer.lap();
    for (0..iterations) |_| {
        std.mem.doNotOptimizeAway(despaceStdlib(bytes, output_buf));
        std.mem.doNotOptimizeAway(bytes);
        std.mem.doNotOptimizeAway(output_buf);
    }
    const stdlib_time = timer.lap();
    _ = despaceAvx512(bytes, output_buf);
    _ = timer.lap();
    for (0..iterations) |_| {
        std.mem.doNotOptimizeAway(despaceAvx512(bytes, output_buf));
        std.mem.doNotOptimizeAway(bytes);
        std.mem.doNotOptimizeAway(output_buf);
    }
    const avx512_time = timer.lap();
    @memcpy(output_buf, bytes);
    _ = timer.lap();
    for (0..iterations) |_| {
        std.mem.doNotOptimizeAway(@memcpy(output_buf, bytes));
        std.mem.doNotOptimizeAway(bytes);
        std.mem.doNotOptimizeAway(output_buf);
    }
    const memcpy_time = timer.lap();
    std.debug.print("string despacing times ({}MB input ran {} times):\n", .{ buffer_size / (1 << 20), iterations });
    std.debug.print("scalar: {}\t\t throughput: {d:.2}GB/s \n", .{ std.fmt.fmtDuration(scalar_time), float(processed_data) / float(scalar_time) });
    std.debug.print("stdlib: {}\t\t throughput: {d:.2}GB/s \n", .{ std.fmt.fmtDuration(stdlib_time), float(processed_data) / float(stdlib_time) });
    std.debug.print("avx512: {}\t\t throughput: {d:.2}GB/s \n", .{ std.fmt.fmtDuration(avx512_time), float(processed_data) / float(avx512_time) });
    std.debug.print("memcpy: {}\t\t throughput: {d:.2}GB/s \n", .{ std.fmt.fmtDuration(memcpy_time), float(processed_data) / float(memcpy_time) });
}

fn float(x: u64) f64 {
    return @floatFromInt(x);
}