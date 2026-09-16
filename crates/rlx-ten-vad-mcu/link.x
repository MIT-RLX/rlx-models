/* Minimal bare-metal layout for QEMU's rv32 `virt` board. A real ESP32-C3
   build uses esp-hal's script instead; only these addresses differ. */
MEMORY {
  RAM : ORIGIN = 0x80000000, LENGTH = 8M
}
ENTRY(_start)
SECTIONS {
  .init  : { KEEP(*(.init)) }        > RAM
  .text  : { *(.text .text.*) }      > RAM
  .rodata : { *(.rodata .rodata.*) } > RAM
  .data  : { *(.data .data.*) *(.sdata .sdata.*) } > RAM
  . = ALIGN(4);
  __bss_start = .;
  .bss (NOLOAD) : { *(.bss .bss.*) *(.sbss .sbss.*) *(COMMON) } > RAM
  . = ALIGN(4);
  __bss_end = .;
  . = ALIGN(16);
  _heap_start = .;
  _stack_top = ORIGIN(RAM) + LENGTH(RAM);
  /DISCARD/ : { *(.eh_frame .eh_frame_hdr) }
}
