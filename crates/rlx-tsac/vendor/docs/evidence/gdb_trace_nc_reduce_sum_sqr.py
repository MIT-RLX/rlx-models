import gdb, struct

# GDB script for tracing nc_reduce_sum_sqr (0x8310) in libnc.so
# 
# USAGE:
#   cd /usr/share/tsac
#   gdb -x /path/to/gdb_trace_nc_reduce_sum_sqr.py /usr/lib/tsac/tsac
#
# This script:
# 1. Disables ASLR for repeatable addresses
# 2. Starts the program (to load libnc.so)
# 3. Reads /proc/<pid>/maps to find libnc.so base
# 4. Sets breakpoint at libnc+0x8310 (nc_reduce_sum_sqr internal)
# 5. Captures first 5 calls + calls #122 (in_proj) and #206 (model.0)

class SqrCapture(gdb.Breakpoint):
    def __init__(self):
        libnc_base = 0
        try:
            pid = gdb.selected_inferior().pid
            with open(f"/proc/{pid}/maps") as f:
                for line in f:
                    if "libnc.so" in line:
                        libnc_base = int(line.split('-')[0], 16)
                        break
        except:
            pass
        
        target = libnc_base + 0x8310
        super().__init__(f"*{hex(target)}", type=gdb.BP_BREAKPOINT)
        self.call_n = 0
        print(f"[GDB] libnc_base={hex(libnc_base)} break_at={hex(target)}")
    
    def stop(self):
        self.call_n += 1
        n = self.call_n
        
        if n > 10 and n not in [122, 206]:
            return False  # Skip most calls
        
        rdi = int(gdb.parse_and_eval("$rdi"))  # data pointer
        rsi = int(gdb.parse_and_eval("$rsi"))  # dim info
        rdx = int(gdb.parse_and_eval("$rdx"))  # count
        rcx = int(gdb.parse_and_eval("$rcx"))  # another count
        
        print(f"\n[GDB Call #{n}] rdi={hex(rdi)} rsi={hex(rsi)} rdx={rdx} rcx={hex(rcx)}")
        
        # Read stack for type parameter (pushed before call)
        try:
            sp = int(gdb.parse_and_eval("$rsp"))
            stack_vals = struct.unpack("<4I", bytes(gdb.selected_inferior().read_memory(sp, 16)))
            print(f"  stack: type={stack_vals[0]} count={stack_vals[1]} dim1={stack_vals[2]} dim2={stack_vals[3]}")
        except:
            pass
        
        # Read first few floats from data buffer
        if rdi > 0x10000 and rdi < 0x7fffffffffff:
            try:
                mem = gdb.selected_inferior().read_memory(rdi, 64)
                floats = struct.unpack("<16f", bytes(mem))
                print(f"  data[0:4]: {[round(f,6) for f in floats[:4]]}")
            except:
                pass
        
        return False  # Don't stop, just log

gdb.execute("set pagination off")
gdb.execute("set disable-randomization on")
gdb.execute("start")
SqrCapture()
gdb.execute("continue")
