#!/usr/bin/env python3
"""
GDB script to trace nc_reduce_sum_sqr internals during BF8 decode.

Usage:
  cd /usr/share/tsac
  gdb -x /path/to/gdb_trace_sqr.py /usr/lib/tsac/tsac

What this does:
1. Breaks at nc_convert call #122 (in_proj.cb0, BF8→float32)
2. Disassembles the nc_convert code to find the actual `call` to sqr
3. Sets a breakpoint at that call target
4. Continues execution — when sqr is hit, dumps:
   - Input data (raw BF8 bytes)
   - Output buffer (accumulated sums)
   - Stack parameters (type, count)
5. Single-steps through the first few instructions of sqr

Why: libnc's nc_convert internal function at file offset 0x8310 
(which computes L2 norms for BF8 groups) cannot be directly 
resolved via symbol. We find it at runtime by disassembling nc_convert.
"""

import gdb, struct, re

gdb.execute("set pagination off")
gdb.execute("set disable-randomization on")

SQR_HITS = 0
CAPTURE_MODE = False  # becomes True when we're at the right nc_convert call

class NcConvertBP(gdb.Breakpoint):
    """Track nc_convert calls to find #122 (in_proj.cb0)"""
    def __init__(self):
        super().__init__("nc_convert", type=gdb.BP_BREAKPOINT)
        self.n = 0
    def stop(self):
        global CAPTURE_MODE
        self.n += 1

        if self.n == 122:
            CAPTURE_MODE = True
            rdi = int(gdb.parse_and_eval("$rdi"))
            print(f"\n=== nc_convert #122 (in_proj.cb0) ===")
            print(f"rdi(tensor)=0x{rdi:016x}")

            # Read tensor info
            nd = int(gdb.parse_and_eval(f"*(int*)((char*){rdi} + 0x64)"))
            d0 = int(gdb.parse_and_eval(f"*(long*)((char*){rdi} + 0xa8)"))
            d1 = int(gdb.parse_and_eval(f"*(long*)((char*){rdi} + 0xb0)")) if nd >= 2 else 1
            d2 = int(gdb.parse_and_eval(f"*(long*)((char*){rdi} + 0xb8)")) if nd >= 3 else 1
            src_type = int(gdb.parse_and_eval(f"*(int*)((char*){rdi} + 0x38)"))
            data_ptr = int(gdb.parse_and_eval(f"*(long*)((char*){rdi} + 0x40)"))

            print(f"dims=[{d0},{d1},{d2}] nd={nd} type={src_type} data=0x{data_ptr:x}")
            if data_ptr > 0x100000:
                raw = bytes(gdb.selected_inferior().read_memory(data_ptr, 24))
                print(f"raw_bytes: {raw.hex()}")

            # nc_convert is at $rip. Disassemble to find call instructions
            rip = int(gdb.parse_and_eval("$rip"))
            print(f"\nnc_convert at: 0x{rip:x}")
            print("Scanning for 'call' instructions in nc_convert...")

            # Disassemble nc_convert to find the call target
            try:
                output = gdb.execute(f"disas 0x{rip:x}, 0x{rip + 0x800:x}", to_string=True)
                # Find `call` instructions that target internal functions
                calls = re.findall(r'call\s+(0x[0-9a-f]+)', output)
                print(f"Found {len(calls)} call targets:")
                for i, target in enumerate(calls[:10]):
                    target_int = int(target, 16)
                    if target_int > 0x100000 and target_int < 0x7fffffffffff:
                        print(f"  call {target}")
                        if i == 0:  # Set BP at first internal call
                            try:
                                gdb.execute(f"break *{target}")
                                print(f"  → BP SET at {target}")
                            except Exception as e:
                                print(f"  → BP ERROR: {e}")
            except Exception as e:
                print(f"disas error: {e}")

            print("Continuing...")
        return False  # Don't stop at nc_convert

class SqrBP(gdb.Breakpoint):
    """Breakpoint at the sqr internal function (set dynamically)"""
    def __init__(self, addr):
        super().__init__(f"*{addr}", type=gdb.BP_BREAKPOINT)
        self.hits = 0
        self.total_input = 0
    def stop(self):
        global SQR_HITS
        self.hits += 1
        SQR_HITS += 1
        h = SQR_HITS

        rdi = int(gdb.parse_and_eval("$rdi"))  # output buffer
        rsi = int(gdb.parse_and_eval("$rsi"))  # ?
        rdx = int(gdb.parse_and_eval("$rdx"))  # data pointer
        rcx = int(gdb.parse_and_eval("$rcx"))
        r8 = int(gdb.parse_and_eval("$r8"))
        r9 = int(gdb.parse_and_eval("$r9"))

        print(f"\n--- SQR call #{h} ---")
        print(f"rdi(buf)=0x{rdi:x} rsi=0x{rsi:x} rdx(data)=0x{rdx:x}")
        print(f"rcx=0x{rcx:x} r8={r8} r9={r9}")

        # Read stack params
        try:
            sp = int(gdb.parse_and_eval("$rsp"))
            stk = bytes(gdb.selected_inferior().read_memory(sp, 24))
            vals = struct.unpack("<6I", stk)
            print(f"stack: type={vals[0]} count={vals[1]} dims=[{vals[2]},{vals[3]}] extra=[{vals[4]},{vals[5]}]")
        except: pass

        # Read output buffer
        if rdi > 0x100000:
            try:
                mem = gdb.selected_inferior().read_memory(rdi, 32)
                floats = struct.unpack("<8f", bytes(mem))
                print(f"buf[0:4]: {[round(f,8) for f in floats[:4]]}")
            except: pass

        # Read input data
        if rdx > 0x100000:
            try:
                mem = gdb.selected_inferior().read_memory(rdx, 32)
                floats = struct.unpack("<8f", bytes(mem))
                print(f"data[0:4]: {[round(f,8) for f in floats[:4]]}")
            except: pass

        # If first call: step through first instructions
        if h <= 2:
            print("Stepping first 5 instructions...")
            for i in range(5):
                gdb.execute("stepi")
            print(f"After stepi: rip=0x{int(gdb.parse_and_eval('$rip')):x}")
            # Show current instruction
            print(gdb.execute("x/i $rip", to_string=True).strip())

        if h >= 3:
            print("3 sqr calls captured. Disabling sqr BP.")
            self.enabled = False
        return False

# Setup
bp1 = NcConvertBP()
gdb.execute("run -v -f d /tmp/short_fast.txc /dev/null")
print(f"\n=== Done. Total sqr calls captured: {SQR_HITS} ===")
