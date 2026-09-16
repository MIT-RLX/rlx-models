/* Deterministic instruction counter for QEMU: mcycle on the `virt` board is
   wall-clock-derived and moves with host load, which makes it useless for
   comparing builds. */
#include <qemu-plugin.h>
#include <inttypes.h>
#include <stdio.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

static uint64_t insns;

static void tb_exec(unsigned int cpu, void *udata) {
    insns += (uint64_t)(uintptr_t)udata;
}

static void tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb *tb) {
    size_t n = qemu_plugin_tb_n_insns(tb);
    qemu_plugin_register_vcpu_tb_exec_cb(tb, tb_exec, QEMU_PLUGIN_CB_NO_REGS,
                                         (void *)(uintptr_t)n);
}

static void at_exit(qemu_plugin_id_t id, void *p) {
    fprintf(stderr, "INSNS %" PRIu64 "\n", insns);
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id,
                                           const qemu_info_t *info,
                                           int argc, char **argv) {
    qemu_plugin_register_vcpu_tb_trans_cb(id, tb_trans);
    qemu_plugin_register_atexit_cb(id, at_exit, NULL);
    return 0;
}
