#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Tests for the hardware additions to llamagputop: CPU temperature on a label-less hwmon
(pre-Zen k10temp), PCIe negotiation and AER counters read from sysfs, the nvidia-smi -q -x
detail feed, and the generic hwmon sensors panel.

Run it next to llamagputop.py:

    python3 -m unittest test_llamagputop_hw

The program under test is the llamagputop.py in this file's own directory. A second copy,
the PRE-CHANGE program, is loaded only when the environment variable LLAMAGPUTOP_BASELINE
names it, and the tests that compare the two ("this change leaves the old output
byte-identical") run only then:

    LLAMAGPUTOP_BASELINE=/path/to/old/llamagputop.py python3 -m unittest test_llamagputop_hw

Without the variable those comparisons are reported as SKIPPED, naming the variable; with
the variable pointing at a file that does not exist they are skipped too, naming the
missing path, never silently run against nothing. Before a merge, run the suite once with
the variable set to the version being replaced (from git history), and expect zero skips.
A frozen copy of the old program is deliberately not shipped with the tests: it would
fail every later legitimate change to the same output.

Every rendering path that existed before is compared between the two on the same input;
the new behaviour is asserted against the program under test alone. Sysfs is never read
directly: every reader is pointed at a tree built in a temporary directory whose layout
mirrors the real one (a PCI node under its root port under pci0000:00, hwmon chips with
the exact attribute names).

The nvidia-smi XML embedded below is a trimmed copy of a real capture from an AMD 990FX
machine (driver 580.173.02, a Tesla P100 and a GeForce RTX 2060): supported_clocks,
processes and accounted_processes are dropped, and serial, uuid and pdi are replaced with
placeholder values. Everything else is as captured, including the sw_power_cap event
reason reported Active on a card in P8 drawing 2 W, which is the negative control for the
throttle row."""
import importlib.machinery
import importlib.util
import inspect
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PROG = os.path.join(HERE, "llamagputop.py")
BASELINE_VAR = "LLAMAGPUTOP_BASELINE"


def _load(name, path):
    # SourceFileLoader takes any file name, so a baseline kept without a .py suffix
    # (a dated copy, say) loads the same way as the program itself
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


L = _load("llamagputop_under_test", PROG)

_baseline = os.environ.get(BASELINE_VAR)
if not _baseline:
    B, NO_BASELINE = None, (f"regression comparison skipped: set {BASELINE_VAR} to the "
                            f"pre-change llamagputop.py")
elif not os.path.isfile(_baseline):
    B, NO_BASELINE = None, f"regression comparison skipped: {BASELINE_VAR}={_baseline} does not exist"
else:
    B, NO_BASELINE = _load("llamagputop_baseline", _baseline), ""
# every test that compares against the pre-change program carries this
needs_baseline = unittest.skipIf(B is None, NO_BASELINE)


# ------------------------------------------------------------------ fixtures
def _w(d, name, content):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name), "w") as f:
        f.write(f"{content}\n")


def _flat(rows):
    return [seg for row in rows for seg in row]


def _text(rows):
    return "\n".join("".join(t for t, _ in row) for row in rows)


def _attrs_of(rows, needle):
    """The attribute of every segment whose text contains `needle`."""
    return [a for t, a in _flat(rows) if needle in t]


# the exact line format of /sys/bus/pci/devices/*/aer_dev_* on the capture machine (kernel
# 6.18): one "Name count" per line, closed by a TOTAL_ERR_* line
AER_COR_ZERO = ("RxErr 0\nBadTLP 0\nBadDLLP 0\nRollover 0\nTimeout 0\nNonFatalErr 0\n"
                "CorrIntErr 0\nHeaderOF 0\nTOTAL_ERR_COR 0")
AER_NONFATAL_ZERO = ("Undefined 0\nDLP 0\nSDES 0\nTLP 0\nFCP 0\nCmpltTO 0\nCmpltAbrt 0\n"
                     "UnxCmplt 0\nRxOF 0\nMalfTLP 0\nECRC 0\nUnsupReq 0\nACSViol 0\n"
                     "TOTAL_ERR_NONFATAL 0")
AER_FATAL_ZERO = AER_NONFATAL_ZERO.replace("TOTAL_ERR_NONFATAL", "TOTAL_ERR_FATAL")
AER_ZERO = {"aer_dev_correctable": AER_COR_ZERO, "aer_dev_nonfatal": AER_NONFATAL_ZERO,
            "aer_dev_fatal": AER_FATAL_ZERO}


def pci_tree(tmp, dev="0000:14:00.0", port="0000:00:04.0",
             cur=("2.5 GT/s PCIe", 4), mx=("8.0 GT/s PCIe", 16),
             port_cur=("2.5 GT/s PCIe", 4), port_max=("5.0 GT/s PCIe", 4), aer=None):
    """<tmp>/pci0000:00/<port>/<dev> with the four link attributes on both nodes, the
    layout pcie_chain's address regex walks. The defaults are the RTX 2060 on the capture
    machine: a gen3 x16 card in a gen2 x4 slot, idling at gen1."""
    p = os.path.join(tmp, "pci0000:00", port)
    d = os.path.join(p, dev)
    for path, c, m in ((d, cur, mx), (p, port_cur, port_max)):
        os.makedirs(path, exist_ok=True)
        if c:
            _w(path, "current_link_speed", c[0])
            _w(path, "current_link_width", c[1])
        if m:
            _w(path, "max_link_speed", m[0])
            _w(path, "max_link_width", m[1])
    for name, content in (aer or {}).items():
        _w(d, name, content)
    return d


class _FakeFeed:
    """The attributes NvidiaGpu.sample reads off _NvidiaFeed, without a thread."""
    def __init__(self, by_index, index_by_bus, detail_by_bus=None, driver=None, detail_ts=0.0):
        self.by_index = by_index
        self.index_by_bus = index_by_bus
        self.detail_by_bus = detail_by_bus or {}
        self.driver = driver or {}
        self.detail_ts = detail_ts
        self.tool_ok = True


def _core(temp):
    """One row of the core --query-gpu table as _NvidiaFeed stores it."""
    return {"name": "x", "util": 10.0, "mem_util": 5.0, "vram_used": 171.0,
            "vram_total": 6144.0, "temp_main": temp, "power": 2.06, "power_cap": 190.0,
            "sclk": 300.0, "mclk": 405.0, "ts": time.monotonic()}


# nvidia-smi -q -x, driver 580.173.02, trimmed and sanitised as described at the top
NVQ_XML = r"""<?xml version="1.0" ?>
<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v13.dtd">
<nvidia_smi_log>
	<timestamp>Tue Sep 15 11:29:55 2026</timestamp>
	<driver_version>580.173.02</driver_version>
	<cuda_version>13.0</cuda_version>
	<attached_gpus>2</attached_gpus>
	<gpu id="00000000:12:00.0">
		<product_name>Tesla P100-PCIE-16GB</product_name>
		<product_brand>Tesla</product_brand>
		<product_architecture>Pascal</product_architecture>
		<display_mode>Requested functionality has been deprecated</display_mode>
		<display_attached>No</display_attached>
		<display_active>Disabled</display_active>
		<persistence_mode>Enabled</persistence_mode>
		<addressing_mode>N/A</addressing_mode>
		<mig_mode>
			<current_mig>N/A</current_mig>
			<pending_mig>N/A</pending_mig>
		</mig_mode>
		<mig_devices>
			None
		</mig_devices>
		<accounting_mode>Disabled</accounting_mode>
		<accounting_mode_buffer_size>4000</accounting_mode_buffer_size>
		<driver_model>
			<current_dm>N/A</current_dm>
			<pending_dm>N/A</pending_dm>
		</driver_model>
		<serial>0000000000001</serial>
		<uuid>GPU-00000000-0000-0000-0000-000000000001</uuid>
		<pdi>0x0</pdi>
		<minor_number>0</minor_number>
		<vbios_version>86.00.52.00.02</vbios_version>
		<multigpu_board>No</multigpu_board>
		<board_id>0x1200</board_id>
		<board_part_number>900-2H400-6300-031</board_part_number>
		<gpu_part_number>15F8-892-A1</gpu_part_number>
		<gpu_fru_part_number>N/A</gpu_fru_part_number>
		<platformInfo>
			<chassis_serial_number>N/A</chassis_serial_number>
			<slot_number>N/A</slot_number>
			<tray_index>N/A</tray_index>
			<host_id>N/A</host_id>
			<peer_type>N/A</peer_type>
			<module_id>1</module_id>
			<gpu_fabric_guid>N/A</gpu_fabric_guid>
		</platformInfo>
		<inforom_version>
			<img_version>H400.0201.00.08</img_version>
			<oem_object>1.1</oem_object>
			<ecc_object>4.1</ecc_object>
			<pwr_object>N/A</pwr_object>
		</inforom_version>
		<inforom_bbx_flush>
			<latest_timestamp>N/A</latest_timestamp>
			<latest_duration>N/A</latest_duration>
		</inforom_bbx_flush>
		<gpu_operation_mode>
			<current_gom>N/A</current_gom>
			<pending_gom>N/A</pending_gom>
		</gpu_operation_mode>
		<c2c_mode>N/A</c2c_mode>
		<gpu_virtualization_mode>
			<virtualization_mode>None</virtualization_mode>
			<host_vgpu_mode>N/A</host_vgpu_mode>
			<vgpu_heterogeneous_mode>N/A</vgpu_heterogeneous_mode>
		</gpu_virtualization_mode>
		<gpu_recovery_action>None</gpu_recovery_action>
		<gsp_firmware_version>N/A</gsp_firmware_version>
		<ibmnpu>
			<relaxed_ordering_mode>N/A</relaxed_ordering_mode>
		</ibmnpu>
		<pci>
			<pci_bus>12</pci_bus>
			<pci_device>00</pci_device>
			<pci_domain>0000</pci_domain>
			<pci_base_class>3</pci_base_class>
			<pci_sub_class>2</pci_sub_class>
			<pci_device_id>15F810DE</pci_device_id>
			<pci_bus_id>00000000:12:00.0</pci_bus_id>
			<pci_sub_system_id>FFFFFFFF</pci_sub_system_id>
			<pci_gpu_link_info>
				<pcie_gen>
					<max_link_gen>2</max_link_gen>
					<current_link_gen>2</current_link_gen>
					<device_current_link_gen>2</device_current_link_gen>
					<max_device_link_gen>3</max_device_link_gen>
					<max_host_link_gen>2</max_host_link_gen>
				</pcie_gen>
				<link_widths>
					<max_link_width>16x</max_link_width>
					<current_link_width>16x</current_link_width>
				</link_widths>
			</pci_gpu_link_info>
			<pci_bridge_chip>
				<bridge_chip_type>N/A</bridge_chip_type>
				<bridge_chip_fw>N/A</bridge_chip_fw>
			</pci_bridge_chip>
			<replay_counter>0</replay_counter>
			<replay_rollover_counter>0</replay_rollover_counter>
			<tx_util>0 KB/s</tx_util>
			<rx_util>0 KB/s</rx_util>
			<atomic_caps_outbound>N/A</atomic_caps_outbound>
			<atomic_caps_inbound>N/A</atomic_caps_inbound>
		</pci>
		<fan_speed>N/A</fan_speed>
		<performance_state>P0</performance_state>
		<clocks_event_reasons>
			<clocks_event_reason_gpu_idle>Active</clocks_event_reason_gpu_idle>
			<clocks_event_reason_applications_clocks_setting>Not Active</clocks_event_reason_applications_clocks_setting>
			<clocks_event_reason_sw_power_cap>Not Active</clocks_event_reason_sw_power_cap>
			<clocks_event_reason_hw_slowdown>Not Active</clocks_event_reason_hw_slowdown>
			<clocks_event_reason_hw_thermal_slowdown>Not Active</clocks_event_reason_hw_thermal_slowdown>
			<clocks_event_reason_hw_power_brake_slowdown>Not Active</clocks_event_reason_hw_power_brake_slowdown>
			<clocks_event_reason_sync_boost>Not Active</clocks_event_reason_sync_boost>
			<clocks_event_reason_sw_thermal_slowdown>Not Active</clocks_event_reason_sw_thermal_slowdown>
			<clocks_event_reason_display_clocks_setting>Not Active</clocks_event_reason_display_clocks_setting>
		</clocks_event_reasons>
		<clocks_event_reasons_counters>
			<clocks_event_reasons_counters_sw_power_cap>427599 us</clocks_event_reasons_counters_sw_power_cap>
			<clocks_event_reasons_counters_sync_boost>0 us</clocks_event_reasons_counters_sync_boost>
			<clocks_event_reasons_counters_sw_therm_slowdown>N/A</clocks_event_reasons_counters_sw_therm_slowdown>
			<clocks_event_reasons_counters_hw_therm_slowdown>0 us</clocks_event_reasons_counters_hw_therm_slowdown>
			<clocks_event_reasons_counters_hw_power_brake>0 us</clocks_event_reasons_counters_hw_power_brake>
		</clocks_event_reasons_counters>
		<sparse_operation_mode>N/A</sparse_operation_mode>
		<fb_memory_usage>
			<total>16384 MiB</total>
			<reserved>115 MiB</reserved>
			<used>5 MiB</used>
			<free>16265 MiB</free>
		</fb_memory_usage>
		<bar1_memory_usage>
			<total>16384 MiB</total>
			<used>2 MiB</used>
			<free>16382 MiB</free>
		</bar1_memory_usage>
		<cc_protected_memory_usage>
			<total>0 MiB</total>
			<used>0 MiB</used>
			<free>0 MiB</free>
		</cc_protected_memory_usage>
		<compute_mode>Default</compute_mode>
		<utilization>
			<gpu_util>0 %</gpu_util>
			<memory_util>0 %</memory_util>
			<encoder_util>0 %</encoder_util>
			<decoder_util>0 %</decoder_util>
			<jpeg_util>N/A</jpeg_util>
			<ofa_util>N/A</ofa_util>
		</utilization>
		<encoder_stats>
			<session_count>0</session_count>
			<average_fps>0</average_fps>
			<average_latency>0</average_latency>
		</encoder_stats>
		<fbc_stats>
			<session_count>0</session_count>
			<average_fps>0</average_fps>
			<average_latency>0</average_latency>
		</fbc_stats>
		<dram_encryption_mode>
			<current_dram_encryption>N/A</current_dram_encryption>
			<pending_dram_encryption>N/A</pending_dram_encryption>
		</dram_encryption_mode>
		<ecc_mode>
			<current_ecc>Enabled</current_ecc>
			<pending_ecc>Enabled</pending_ecc>
		</ecc_mode>
		<ecc_errors>
			<volatile>
				<single_bit>
					<device_memory>0</device_memory>
					<register_file>0</register_file>
					<l1_cache>N/A</l1_cache>
					<l2_cache>0</l2_cache>
					<texture_memory>0</texture_memory>
					<texture_shm>0</texture_shm>
					<cbu>N/A</cbu>
					<total>0</total>
				</single_bit>
				<double_bit>
					<device_memory>0</device_memory>
					<register_file>0</register_file>
					<l1_cache>N/A</l1_cache>
					<l2_cache>0</l2_cache>
					<texture_memory>0</texture_memory>
					<texture_shm>0</texture_shm>
					<cbu>N/A</cbu>
					<total>0</total>
				</double_bit>
			</volatile>
			<aggregate>
				<single_bit>
					<device_memory>0</device_memory>
					<register_file>0</register_file>
					<l1_cache>N/A</l1_cache>
					<l2_cache>0</l2_cache>
					<texture_memory>0</texture_memory>
					<texture_shm>0</texture_shm>
					<cbu>N/A</cbu>
					<total>0</total>
				</single_bit>
				<double_bit>
					<device_memory>0</device_memory>
					<register_file>0</register_file>
					<l1_cache>N/A</l1_cache>
					<l2_cache>0</l2_cache>
					<texture_memory>0</texture_memory>
					<texture_shm>0</texture_shm>
					<cbu>N/A</cbu>
					<total>0</total>
				</double_bit>
			</aggregate>
		</ecc_errors>
		<retired_pages>
			<multiple_single_bit_retirement>
				<retired_count>0</retired_count>
				<retired_pagelist>
				</retired_pagelist>
			</multiple_single_bit_retirement>
			<double_bit_retirement>
				<retired_count>0</retired_count>
				<retired_pagelist>
				</retired_pagelist>
			</double_bit_retirement>
			<pending_blacklist>No</pending_blacklist>
			<pending_retirement>No</pending_retirement>
		</retired_pages>
		<remapped_rows>N/A</remapped_rows>
		<temperature>
			<gpu_temp>38 C</gpu_temp>
			<gpu_temp_tlimit>N/A</gpu_temp_tlimit>
			<gpu_temp_max_threshold>85 C</gpu_temp_max_threshold>
			<gpu_temp_slow_threshold>82 C</gpu_temp_slow_threshold>
			<gpu_temp_max_gpu_threshold>N/A</gpu_temp_max_gpu_threshold>
			<gpu_target_temperature>N/A</gpu_target_temperature>
			<memory_temp>N/A</memory_temp>
			<gpu_temp_max_mem_threshold>N/A</gpu_temp_max_mem_threshold>
		</temperature>
		<supported_gpu_target_temp>
			<gpu_target_temp_min>N/A</gpu_target_temp_min>
			<gpu_target_temp_max>N/A</gpu_target_temp_max>
		</supported_gpu_target_temp>
		<gpu_power_readings>
			<power_state>P0</power_state>
			<average_power_draw>N/A</average_power_draw>
			<instant_power_draw>25.53 W</instant_power_draw>
			<current_power_limit>250.00 W</current_power_limit>
			<requested_power_limit>250.00 W</requested_power_limit>
			<default_power_limit>250.00 W</default_power_limit>
			<min_power_limit>125.00 W</min_power_limit>
			<max_power_limit>250.00 W</max_power_limit>
		</gpu_power_readings>
		<gpu_memory_power_readings>
			<average_power_draw>N/A</average_power_draw>
			<instant_power_draw>N/A</instant_power_draw>
		</gpu_memory_power_readings>
		<module_power_readings>
			<power_state>P0</power_state>
			<average_power_draw>N/A</average_power_draw>
			<instant_power_draw>N/A</instant_power_draw>
			<current_power_limit>N/A</current_power_limit>
			<requested_power_limit>N/A</requested_power_limit>
			<default_power_limit>N/A</default_power_limit>
			<min_power_limit>N/A</min_power_limit>
			<max_power_limit>N/A</max_power_limit>
		</module_power_readings>
		<power_smoothing>N/A</power_smoothing>
		<power_profiles>
			<power_profile_requested_profiles>N/A</power_profile_requested_profiles>
			<power_profile_enforced_profiles>N/A</power_profile_enforced_profiles>
		</power_profiles>
		<clocks>
			<graphics_clock>405 MHz</graphics_clock>
			<sm_clock>405 MHz</sm_clock>
			<mem_clock>715 MHz</mem_clock>
			<video_clock>835 MHz</video_clock>
		</clocks>
		<applications_clocks>
			<graphics_clock>1189 MHz</graphics_clock>
			<mem_clock>715 MHz</mem_clock>
		</applications_clocks>
		<default_applications_clocks>
			<graphics_clock>1189 MHz</graphics_clock>
			<mem_clock>715 MHz</mem_clock>
		</default_applications_clocks>
		<deferred_clocks>
			<mem_clock>N/A</mem_clock>
		</deferred_clocks>
		<max_clocks>
			<graphics_clock>1328 MHz</graphics_clock>
			<sm_clock>1328 MHz</sm_clock>
			<mem_clock>715 MHz</mem_clock>
			<video_clock>1328 MHz</video_clock>
		</max_clocks>
		<max_customer_boost_clocks>
			<graphics_clock>1328 MHz</graphics_clock>
		</max_customer_boost_clocks>
		<clock_policy>
			<auto_boost>N/A</auto_boost>
			<auto_boost_default>N/A</auto_boost_default>
		</clock_policy>
		<fabric>
			<state>N/A</state>
			<status>N/A</status>
			<cliqueId>N/A</cliqueId>
			<clusterUuid>N/A</clusterUuid>
			<health>
				<summary>N/A</summary>
				<bandwidth>N/A</bandwidth>
				<route_recovery_in_progress>N/A</route_recovery_in_progress>
				<route_unhealthy>N/A</route_unhealthy>
				<access_timeout_recovery>N/A</access_timeout_recovery>
				<incorrect_configuration>N/A</incorrect_configuration>
				<partition_assigned>N/A</partition_assigned>
			</health>
		</fabric>
		<capabilities>
			<egm>disabled</egm>
		</capabilities>
	</gpu>

	<gpu id="00000000:14:00.0">
		<product_name>NVIDIA GeForce RTX 2060</product_name>
		<product_brand>GeForce</product_brand>
		<product_architecture>Turing</product_architecture>
		<display_mode>Requested functionality has been deprecated</display_mode>
		<display_attached>Yes</display_attached>
		<display_active>Enabled</display_active>
		<persistence_mode>Enabled</persistence_mode>
		<addressing_mode>None</addressing_mode>
		<mig_mode>
			<current_mig>N/A</current_mig>
			<pending_mig>N/A</pending_mig>
		</mig_mode>
		<mig_devices>
			None
		</mig_devices>
		<accounting_mode>Disabled</accounting_mode>
		<accounting_mode_buffer_size>4000</accounting_mode_buffer_size>
		<driver_model>
			<current_dm>N/A</current_dm>
			<pending_dm>N/A</pending_dm>
		</driver_model>
		<serial>0000000000002</serial>
		<uuid>GPU-00000000-0000-0000-0000-000000000002</uuid>
		<pdi>0x0</pdi>
		<minor_number>1</minor_number>
		<vbios_version>90.06.2E.40.56</vbios_version>
		<multigpu_board>No</multigpu_board>
		<board_id>0x1400</board_id>
		<board_part_number>N/A</board_part_number>
		<gpu_part_number>1F08-200-A1</gpu_part_number>
		<gpu_fru_part_number>N/A</gpu_fru_part_number>
		<platformInfo>
			<chassis_serial_number>N/A</chassis_serial_number>
			<slot_number>N/A</slot_number>
			<tray_index>N/A</tray_index>
			<host_id>N/A</host_id>
			<peer_type>N/A</peer_type>
			<module_id>1</module_id>
			<gpu_fabric_guid>N/A</gpu_fabric_guid>
		</platformInfo>
		<inforom_version>
			<img_version>G001.0000.02.04</img_version>
			<oem_object>1.1</oem_object>
			<ecc_object>N/A</ecc_object>
			<pwr_object>N/A</pwr_object>
		</inforom_version>
		<inforom_bbx_flush>
			<latest_timestamp>N/A</latest_timestamp>
			<latest_duration>N/A</latest_duration>
		</inforom_bbx_flush>
		<gpu_operation_mode>
			<current_gom>N/A</current_gom>
			<pending_gom>N/A</pending_gom>
		</gpu_operation_mode>
		<c2c_mode>N/A</c2c_mode>
		<gpu_virtualization_mode>
			<virtualization_mode>None</virtualization_mode>
			<host_vgpu_mode>N/A</host_vgpu_mode>
			<vgpu_heterogeneous_mode>N/A</vgpu_heterogeneous_mode>
		</gpu_virtualization_mode>
		<gpu_recovery_action>None</gpu_recovery_action>
		<gsp_firmware_version>580.173.02</gsp_firmware_version>
		<ibmnpu>
			<relaxed_ordering_mode>N/A</relaxed_ordering_mode>
		</ibmnpu>
		<pci>
			<pci_bus>14</pci_bus>
			<pci_device>00</pci_device>
			<pci_domain>0000</pci_domain>
			<pci_base_class>3</pci_base_class>
			<pci_sub_class>0</pci_sub_class>
			<pci_device_id>1F0810DE</pci_device_id>
			<pci_bus_id>00000000:14:00.0</pci_bus_id>
			<pci_sub_system_id>37D41458</pci_sub_system_id>
			<pci_gpu_link_info>
				<pcie_gen>
					<max_link_gen>2</max_link_gen>
					<current_link_gen>1</current_link_gen>
					<device_current_link_gen>1</device_current_link_gen>
					<max_device_link_gen>3</max_device_link_gen>
					<max_host_link_gen>2</max_host_link_gen>
				</pcie_gen>
				<link_widths>
					<max_link_width>16x</max_link_width>
					<current_link_width>4x</current_link_width>
				</link_widths>
			</pci_gpu_link_info>
			<pci_bridge_chip>
				<bridge_chip_type>N/A</bridge_chip_type>
				<bridge_chip_fw>N/A</bridge_chip_fw>
			</pci_bridge_chip>
			<replay_counter>0</replay_counter>
			<replay_rollover_counter>0</replay_rollover_counter>
			<tx_util>683 KB/s</tx_util>
			<rx_util>439 KB/s</rx_util>
			<atomic_caps_outbound>N/A</atomic_caps_outbound>
			<atomic_caps_inbound>N/A</atomic_caps_inbound>
		</pci>
		<fan_speed>0 %</fan_speed>
		<performance_state>P8</performance_state>
		<clocks_event_reasons>
			<clocks_event_reason_gpu_idle>Not Active</clocks_event_reason_gpu_idle>
			<clocks_event_reason_applications_clocks_setting>Not Active</clocks_event_reason_applications_clocks_setting>
			<clocks_event_reason_sw_power_cap>Active</clocks_event_reason_sw_power_cap>
			<clocks_event_reason_hw_slowdown>Not Active</clocks_event_reason_hw_slowdown>
			<clocks_event_reason_hw_thermal_slowdown>Not Active</clocks_event_reason_hw_thermal_slowdown>
			<clocks_event_reason_hw_power_brake_slowdown>Not Active</clocks_event_reason_hw_power_brake_slowdown>
			<clocks_event_reason_sync_boost>Not Active</clocks_event_reason_sync_boost>
			<clocks_event_reason_sw_thermal_slowdown>Not Active</clocks_event_reason_sw_thermal_slowdown>
			<clocks_event_reason_display_clocks_setting>Not Active</clocks_event_reason_display_clocks_setting>
		</clocks_event_reasons>
		<clocks_event_reasons_counters>
			<clocks_event_reasons_counters_sw_power_cap>4991356120 us</clocks_event_reasons_counters_sw_power_cap>
			<clocks_event_reasons_counters_sync_boost>0 us</clocks_event_reasons_counters_sync_boost>
			<clocks_event_reasons_counters_sw_therm_slowdown>0 us</clocks_event_reasons_counters_sw_therm_slowdown>
			<clocks_event_reasons_counters_hw_therm_slowdown>0 us</clocks_event_reasons_counters_hw_therm_slowdown>
			<clocks_event_reasons_counters_hw_power_brake>0 us</clocks_event_reasons_counters_hw_power_brake>
		</clocks_event_reasons_counters>
		<sparse_operation_mode>N/A</sparse_operation_mode>
		<fb_memory_usage>
			<total>6144 MiB</total>
			<reserved>409 MiB</reserved>
			<used>171 MiB</used>
			<free>5566 MiB</free>
		</fb_memory_usage>
		<bar1_memory_usage>
			<total>256 MiB</total>
			<used>14 MiB</used>
			<free>242 MiB</free>
		</bar1_memory_usage>
		<cc_protected_memory_usage>
			<total>0 MiB</total>
			<used>0 MiB</used>
			<free>0 MiB</free>
		</cc_protected_memory_usage>
		<compute_mode>Default</compute_mode>
		<utilization>
			<gpu_util>10 %</gpu_util>
			<memory_util>5 %</memory_util>
			<encoder_util>0 %</encoder_util>
			<decoder_util>0 %</decoder_util>
			<jpeg_util>0 %</jpeg_util>
			<ofa_util>0 %</ofa_util>
		</utilization>
		<encoder_stats>
			<session_count>0</session_count>
			<average_fps>0</average_fps>
			<average_latency>0</average_latency>
		</encoder_stats>
		<fbc_stats>
			<session_count>0</session_count>
			<average_fps>0</average_fps>
			<average_latency>0</average_latency>
		</fbc_stats>
		<dram_encryption_mode>
			<current_dram_encryption>N/A</current_dram_encryption>
			<pending_dram_encryption>N/A</pending_dram_encryption>
		</dram_encryption_mode>
		<ecc_mode>
			<current_ecc>N/A</current_ecc>
			<pending_ecc>N/A</pending_ecc>
		</ecc_mode>
		<ecc_errors>
			<volatile>
				<sram_correctable>N/A</sram_correctable>
				<sram_uncorrectable>N/A</sram_uncorrectable>
				<dram_correctable>N/A</dram_correctable>
				<dram_uncorrectable>N/A</dram_uncorrectable>
			</volatile>
			<aggregate>
				<sram_correctable>N/A</sram_correctable>
				<sram_uncorrectable>N/A</sram_uncorrectable>
				<dram_correctable>N/A</dram_correctable>
				<dram_uncorrectable>N/A</dram_uncorrectable>
			</aggregate>
		</ecc_errors>
		<retired_pages>
			<multiple_single_bit_retirement>
				<retired_count>N/A</retired_count>
				<retired_pagelist>N/A</retired_pagelist>
			</multiple_single_bit_retirement>
			<double_bit_retirement>
				<retired_count>N/A</retired_count>
				<retired_pagelist>N/A</retired_pagelist>
			</double_bit_retirement>
			<pending_blacklist>N/A</pending_blacklist>
			<pending_retirement>N/A</pending_retirement>
		</retired_pages>
		<remapped_rows>N/A</remapped_rows>
		<temperature>
			<gpu_temp>32 C</gpu_temp>
			<gpu_temp_tlimit>N/A</gpu_temp_tlimit>
			<gpu_temp_max_threshold>93 C</gpu_temp_max_threshold>
			<gpu_temp_slow_threshold>90 C</gpu_temp_slow_threshold>
			<gpu_temp_max_gpu_threshold>88 C</gpu_temp_max_gpu_threshold>
			<gpu_target_temperature>83 C</gpu_target_temperature>
			<memory_temp>N/A</memory_temp>
			<gpu_temp_max_mem_threshold>N/A</gpu_temp_max_mem_threshold>
		</temperature>
		<supported_gpu_target_temp>
			<gpu_target_temp_min>65 C</gpu_target_temp_min>
			<gpu_target_temp_max>87 C</gpu_target_temp_max>
		</supported_gpu_target_temp>
		<gpu_power_readings>
			<power_state>P8</power_state>
			<average_power_draw>N/A</average_power_draw>
			<instant_power_draw>2.06 W</instant_power_draw>
			<current_power_limit>190.00 W</current_power_limit>
			<requested_power_limit>190.00 W</requested_power_limit>
			<default_power_limit>190.00 W</default_power_limit>
			<min_power_limit>125.00 W</min_power_limit>
			<max_power_limit>240.00 W</max_power_limit>
		</gpu_power_readings>
		<gpu_memory_power_readings>
			<average_power_draw>N/A</average_power_draw>
			<instant_power_draw>N/A</instant_power_draw>
		</gpu_memory_power_readings>
		<module_power_readings>
			<power_state>P8</power_state>
			<average_power_draw>N/A</average_power_draw>
			<instant_power_draw>N/A</instant_power_draw>
			<current_power_limit>N/A</current_power_limit>
			<requested_power_limit>N/A</requested_power_limit>
			<default_power_limit>N/A</default_power_limit>
			<min_power_limit>N/A</min_power_limit>
			<max_power_limit>N/A</max_power_limit>
		</module_power_readings>
		<power_smoothing>N/A</power_smoothing>
		<power_profiles>
			<power_profile_requested_profiles>N/A</power_profile_requested_profiles>
			<power_profile_enforced_profiles>N/A</power_profile_enforced_profiles>
		</power_profiles>
		<clocks>
			<graphics_clock>300 MHz</graphics_clock>
			<sm_clock>300 MHz</sm_clock>
			<mem_clock>405 MHz</mem_clock>
			<video_clock>540 MHz</video_clock>
		</clocks>
		<applications_clocks>
			<graphics_clock>N/A</graphics_clock>
			<mem_clock>N/A</mem_clock>
		</applications_clocks>
		<default_applications_clocks>
			<graphics_clock>N/A</graphics_clock>
			<mem_clock>N/A</mem_clock>
		</default_applications_clocks>
		<deferred_clocks>
			<mem_clock>N/A</mem_clock>
		</deferred_clocks>
		<max_clocks>
			<graphics_clock>2175 MHz</graphics_clock>
			<sm_clock>2175 MHz</sm_clock>
			<mem_clock>7071 MHz</mem_clock>
			<video_clock>2025 MHz</video_clock>
		</max_clocks>
		<max_customer_boost_clocks>
			<graphics_clock>N/A</graphics_clock>
		</max_customer_boost_clocks>
		<clock_policy>
			<auto_boost>N/A</auto_boost>
			<auto_boost_default>N/A</auto_boost_default>
		</clock_policy>
		<fabric>
			<state>N/A</state>
			<status>N/A</status>
			<cliqueId>N/A</cliqueId>
			<clusterUuid>N/A</clusterUuid>
			<health>
				<summary>N/A</summary>
				<bandwidth>N/A</bandwidth>
				<route_recovery_in_progress>N/A</route_recovery_in_progress>
				<route_unhealthy>N/A</route_unhealthy>
				<access_timeout_recovery>N/A</access_timeout_recovery>
				<incorrect_configuration>N/A</incorrect_configuration>
				<partition_assigned>N/A</partition_assigned>
			</health>
		</fabric>
		<capabilities>
			<egm>disabled</egm>
		</capabilities>
	</gpu>
</nvidia_smi_log>
"""


_MISSING = object()


class _Base(unittest.TestCase):
    def setUp(self):
        # distinct integers so an attribute can be asserted; the modules start at 0 for
        # all five, which would make OK, WARN and CRIT indistinguishable
        # B is None without a baseline (see the module docstring)
        self._saved = [(m, m.OK, m.WARN, m.CRIT, m.DIM, m.SER) for m in (L, B) if m is not None]
        for m, *_ in self._saved:
            m.OK, m.WARN, m.CRIT, m.DIM, m.SER = 11, 12, 13, 14, 15
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # AER liveness is decided from two sysfs directories (batch 3); every test gets
        # empty fakes for both, so no verdict depends on the machine the suite runs on
        # (this one has no AER service). A class that needs a live root port adds the
        # <rootport>:pcie002 entry in its own setUp.
        self._dirs = [(n, getattr(L, n, _MISSING)) for n in ("_PCIE_SERVICES_DIR", "_ACPI_TABLES_DIR")]
        for n, _ in self._dirs:
            d = os.path.join(self.tmp.name, n.strip("_").lower())
            os.makedirs(d)
            setattr(L, n, d)
        # the details menu state (batch 5) is module level; a test that flips it must
        # not leak into the next
        self._menu_state = [(n, dict(getattr(L, n))) for n in ("_details", "_menu")
                            if isinstance(getattr(L, n, None), dict)]

    def tearDown(self):
        for m, *vals in self._saved:
            m.OK, m.WARN, m.CRIT, m.DIM, m.SER = vals
        for n, old in self._dirs:
            if old is _MISSING:
                delattr(L, n)
            else:
                setattr(L, n, old)
        for n, saved in self._menu_state:
            getattr(L, n).clear()
            getattr(L, n).update(saved)

    def need(self, name):
        """A new function: fail (not error) with a readable message while it is missing."""
        fn = getattr(L, name, None)
        self.assertIsNotNone(fn, f"behaviour missing: {name}")
        return fn

    def assertNoWarnCrit(self, rows):
        bad = [(t, a) for t, a in _flat(rows) if a in (L.WARN, L.CRIT)]
        self.assertEqual(bad, [], "WARN/CRIT attribute on a healthy reading")


# ================================================================ A. CPU temperature
class CpuTemps(_Base):
    def _hw(self, name, files, sub="hwmon1"):
        d = os.path.join(self.tmp.name, sub)
        _w(d, "name", name)
        for k, v in files.items():
            _w(d, k, v)
        return d

    def _cpu(self, t):
        return {"name": "AMD FX-8350", "ncpu": 8, "util": 10.0, "temp": t, "temps": {"Tctl": t},
                "freq": 4000, "power": None, "rapl_present": False,
                "load": ["0.1", "0.2", "0.3"], "loadavg": "0.1",
                "temp_limits": {"Tctl": (70, 80)}, "temp_relative": ["Tctl"]}

    def test_prezen_k10temp_unlabeled_channel_is_tctl(self):
        # this machine: k10temp on an FX-8350 exposes temp1_input/max/crit and no label
        hw = self._hw("k10temp", {"temp1_input": 38000, "temp1_max": 70000,
                                  "temp1_crit": 80000, "temp1_crit_hyst": 78000})
        un = self.need("hwmon_unlabeled_temps")(hw)
        self.assertEqual(un, {"Tctl": "temp1"})
        self.assertEqual(self.need("hwmon_temp_limits")(hw, un), {"Tctl": (70, 80)})

    def test_zen_labeled_channels_are_not_unlabeled(self):
        hw = self._hw("k10temp", {"temp1_input": 45000, "temp1_label": "Tctl",
                                  "temp3_input": 44000, "temp3_label": "Tccd1"})
        self.assertEqual(self.need("hwmon_unlabeled_temps")(hw), {})

    def test_generic_chip_keeps_the_channel_name(self):
        hw = self._hw("nct6775", {"temp2_input": 30000})
        self.assertEqual(self.need("hwmon_unlabeled_temps")(hw), {"temp2": "temp2"})

    def test_no_hwmon_gives_empty(self):
        self.assertEqual(self.need("hwmon_unlabeled_temps")(None), {})
        self.assertEqual(self.need("hwmon_temp_limits")(None, {"Tctl": "temp1"}), {})

    def test_limits_only_when_the_driver_declares_them(self):
        hw = self._hw("nct6775", {"temp2_input": 30000})
        self.assertEqual(self.need("hwmon_temp_limits")(hw, {"temp2": "temp2"}), {})
        hw2 = self._hw("nct6775", {"temp2_input": 30000, "temp2_crit": 95000}, sub="hwmon2")
        self.assertEqual(self.need("hwmon_temp_limits")(hw2, {"temp2": "temp2"}), {"temp2": (0, 95)})

    def test_cpu_rows_colour_by_limits_and_mark_relative(self):
        rows = L._cpu_rows(self._cpu(72), 120)
        self.assertIn("Tctl 72°C", _text(rows))
        self.assertIn("rel", _text(rows))
        self.assertEqual(_attrs_of(rows, "72°C"), [L.WARN])
        self.assertEqual(_attrs_of(L._cpu_rows(self._cpu(81), 120), "81°C"), [L.CRIT])
        self.assertEqual(_attrs_of(L._cpu_rows(self._cpu(50), 120), "50°C"), [L.OK])

    def _cpu_without_limits(self):
        c = self._cpu(72)
        del c["temp_limits"]
        del c["temp_relative"]
        return c

    def test_cpu_rows_without_limits_carry_no_marker(self):
        c = self._cpu_without_limits()
        self.assertNotIn("rel", _text(L._cpu_rows(c, 120)))

    @needs_baseline
    def test_cpu_rows_without_limits_match_the_backup(self):
        c = self._cpu_without_limits()
        self.assertEqual(L._cpu_rows(c, 120), B._cpu_rows(c, 120))

    def test_cpu_sample_carries_limits_and_relative(self):
        c = L.cpu_sample()
        self.assertIn("temp_limits", c, "behaviour missing: cpu_sample temp_limits")
        self.assertIn("temp_relative", c, "behaviour missing: cpu_sample temp_relative")
        self.assertIsInstance(c["temp_limits"], dict)
        self.assertIsInstance(c["temp_relative"], list)


# ================================================================ B. PCIe negotiation
class PcieNegotiation(_Base):
    def test_2060_like_link_in_a_narrow_slow_slot(self):
        d = pci_tree(self.tmp.name)
        n = self.need("_pcie_negotiation")(d)
        self.assertEqual((n.get("gen"), n.get("card_max_gen"), n.get("card_max_width"),
                          n.get("slot_max_gen"), n.get("slot_max_width")), (1, 3, 16, 2, 4))
        self.assertEqual((n.get("card_gts"), n.get("card_width"),
                          n.get("card_max_gts"), n.get("slot_max_gts")), (2.5, 4, 8.0, 5.0))
        self.assertTrue(n.get("width_limited_by_slot"))
        self.assertTrue(n.get("speed_limited_by_host"))
        self.assertFalse(n.get("width_degraded"))
        self.assertTrue(n.get("speed_below_both"))
        self.assertIsNone(n.get("aer"))
        self.assertIn("card_gts", L.pcie_chain(d))

    @needs_baseline
    def test_2060_like_link_keeps_the_old_pcie_chain_keys(self):
        # the keys pcie_chain always had keep their values; the new ones ride along
        d = pci_tree(self.tmp.name)
        link, orig = L.pcie_chain(d), B.pcie_chain(d)
        self.assertEqual({k: link[k] for k in orig}, orig)
        self.assertEqual(L.pcie_text(link), B.pcie_text(orig))

    def test_healthy_p100_has_no_flags_and_no_warning(self):
        d = pci_tree(self.tmp.name, dev="0000:12:00.0", port="0000:00:02.0",
                     cur=("5.0 GT/s PCIe", 16), mx=("8.0 GT/s PCIe", 16),
                     port_cur=("5.0 GT/s PCIe", 16), port_max=("5.0 GT/s PCIe", 16),
                     aer=AER_ZERO)
        n = self.need("_pcie_negotiation")(d)
        self.assertFalse(n.get("width_degraded"))
        self.assertFalse(n.get("speed_below_both"))
        self.assertFalse(n.get("width_limited_by_slot"))
        self.assertTrue(n.get("speed_limited_by_host"))     # gen3 card, gen2 root port
        rows = self.need("_pcie_rows")({"pcie": L.pcie_chain(d)}, 100)
        self.assertIn("gen2 x16 now", _text(rows))
        self.assertIn("card gen3 x16", _text(rows))
        self.assertIn("slot gen2 x16", _text(rows))
        self.assertNotIn("⚠", _text(rows))
        self.assertNoWarnCrit(rows)

    def test_degraded_width_is_flagged_and_warned(self):
        d = pci_tree(self.tmp.name, dev="0000:12:00.0", port="0000:00:02.0",
                     cur=("5.0 GT/s PCIe", 8), mx=("8.0 GT/s PCIe", 16),
                     port_cur=("5.0 GT/s PCIe", 8), port_max=("5.0 GT/s PCIe", 16))
        n = self.need("_pcie_negotiation")(d)
        self.assertTrue(n.get("width_degraded"))
        self.assertFalse(n.get("width_limited_by_slot"))
        rows = self.need("_pcie_rows")({"pcie": L.pcie_chain(d)}, 100)
        self.assertIn("⚠", _text(rows))
        self.assertIn("x8 below x16", _text(rows))
        self.assertIn(L.WARN, [a for _, a in _flat(rows)])

    def test_card_without_a_pci_parent_has_no_slot(self):
        d = os.path.join(self.tmp.name, "pci0000:00", "0000:03:00.0")
        _w(d, "current_link_speed", "8.0 GT/s PCIe")
        _w(d, "current_link_width", 16)
        _w(d, "max_link_speed", "16.0 GT/s PCIe")
        _w(d, "max_link_width", 16)
        n = self.need("_pcie_negotiation")(d)
        self.assertEqual((n.get("gen"), n.get("card_max_gen")), (3, 4))
        self.assertIsNone(n.get("slot_max_gts"))
        self.assertIsNone(n.get("slot_max_gen"))
        self.assertFalse(n.get("width_limited_by_slot"))
        self.assertFalse(n.get("speed_limited_by_host"))
        self.assertTrue(n.get("speed_below_both"))          # 8.0 < the card's own 16.0
        rows = self.need("_pcie_rows")({"pcie": L.pcie_chain(d)}, 100)
        self.assertNotIn("slot", _text(rows))

    def test_device_without_link_attributes(self):
        d = os.path.join(self.tmp.name, "pci0000:00", "0000:00:04.0", "0000:14:00.0")
        os.makedirs(d)
        self.assertEqual(self.need("_pcie_negotiation")(d), {})
        self.assertEqual(self.need("_pcie_rows")({"pcie": None}, 100), [])
        self.assertEqual(self.need("_pcie_rows")({}, 100), [])
        old = {"gts": 5.0, "width": 16, "gbs": 8.0, "narrow_gbs": 8.0, "bottleneck": False,
               "narrow_gts": 5.0, "narrow_width": 16}
        self.assertEqual(self.need("_pcie_rows")({"pcie": old}, 100), [])

    def test_nvidia_extras_and_driver_disagreement(self):
        d = pci_tree(self.tmp.name)
        s = {"pcie": L.pcie_chain(d), "pcie_tx_kbs": 683, "pcie_rx_kbs": 439,
             "pcie_replay": 0, "pcie_replay_rollover": 0,
             "nv_gen_dev_max": 3, "nv_gen_host_max": 2}
        rows = self.need("_pcie_rows")(s, 120)
        t = _text(rows)
        self.assertIn("tx 683 KB/s", t)
        self.assertIn("rx 439 KB/s", t)
        self.assertIn("replays 0", t)
        self.assertIn("rollovers 0", t)
        self.assertNotIn("driver reports", t)
        self.assertNoWarnCrit(rows)
        s["nv_gen_host_max"] = 3
        rows = self.need("_pcie_rows")(s, 120)
        self.assertIn("driver reports card gen3 host gen3", _text(rows))
        self.assertIn(L.WARN, _attrs_of(rows, "driver reports"))


class Aer(_Base):
    def setUp(self):
        # batch 3: the figures these tests assert are drawn only while AER is live, so
        # the fake root port (pci_tree's default 0000:00:04.0) gets its AER service
        super().setUp()
        _w(L._PCIE_SERVICES_DIR, "0000:00:04.0:pcie002", "")

    def _tree(self, cor, nonfatal=AER_NONFATAL_ZERO, fatal=AER_FATAL_ZERO):
        return pci_tree(self.tmp.name, aer={"aer_dev_correctable": cor,
                                            "aer_dev_nonfatal": nonfatal,
                                            "aer_dev_fatal": fatal})

    def test_first_read_baselines_then_counts_the_session(self):
        d = self._tree("RxErr 3\nBadTLP 0\nTOTAL_ERR_COR 3")
        a = self.need("pcie_aer")(d)
        self.assertEqual(a.get("correctable"), 3)
        self.assertEqual(a.get("nonfatal"), 0)
        self.assertEqual(a.get("fatal"), 0)
        self.assertIn("RxErr 3", a.get("nonzero") or [])
        self.assertEqual(a.get("correctable_session"), 0)
        rows = self.need("_pcie_rows")({"pcie": L.pcie_chain(d)}, 120)
        self.assertIn("corrected 3", _text(rows))
        self.assertNoWarnCrit(rows)             # errors that predate the session
        _w(d, "aer_dev_correctable", "RxErr 5\nBadTLP 0\nTOTAL_ERR_COR 5")
        a = self.need("pcie_aer")(d)
        self.assertEqual(a.get("correctable_session"), 2)
        rows = self.need("_pcie_rows")({"pcie": L.pcie_chain(d)}, 120)
        self.assertIn("+2 this session", _text(rows))
        self.assertIn(L.WARN, _attrs_of(rows, "this session"))
        self.assertNotIn(L.CRIT, [a for _, a in _flat(rows)])

    def test_counter_reset_rebaselines(self):
        d = self._tree("RxErr 5\nTOTAL_ERR_COR 5")
        self.need("pcie_aer")(d)
        _w(d, "aer_dev_correctable", "RxErr 1\nTOTAL_ERR_COR 1")
        a = self.need("pcie_aer")(d)
        self.assertEqual(a.get("correctable_session"), 0)
        _w(d, "aer_dev_correctable", "RxErr 2\nTOTAL_ERR_COR 2")
        self.assertEqual(self.need("pcie_aer")(d).get("correctable_session"), 1)

    def test_all_zero_is_quiet(self):
        d = self._tree(AER_COR_ZERO)
        a = self.need("pcie_aer")(d)
        self.assertEqual(a.get("nonzero"), [])
        self.assertEqual((a.get("correctable"), a.get("correctable_session")), (0, 0))
        rows = self.need("_pcie_rows")({"pcie": L.pcie_chain(d)}, 120)
        self.assertIn("aer", _text(rows))
        self.assertNoWarnCrit(rows)

    def test_uncorrectable_is_crit(self):
        d = self._tree(AER_COR_ZERO, nonfatal="Undefined 0\nMalfTLP 1\nTOTAL_ERR_NONFATAL 1")
        a = self.need("pcie_aer")(d)
        self.assertEqual(a.get("nonfatal"), 1)
        self.assertIn("MalfTLP 1", a.get("nonzero") or [])
        rows = self.need("_pcie_rows")({"pcie": L.pcie_chain(d)}, 120)
        self.assertIn(L.CRIT, _attrs_of(rows, "1"))
        self.assertIn("MalfTLP 1", _text(rows))

    def test_no_aer_files(self):
        d = pci_tree(self.tmp.name)
        self.assertIsNone(self.need("pcie_aer")(d))
        rows = self.need("_pcie_rows")({"pcie": L.pcie_chain(d)}, 120)
        self.assertNotIn("aer", _text(rows))


# ================================================================ C. NVIDIA detail
def _gpu(tc, bus):
    """One card's parsed detail; goes through need() so a missing parser FAILS, not errors."""
    return tc.need("parse_nvidia_smi_xml")(NVQ_XML)["gpus"][bus]


class NvidiaXml(_Base):
    def test_bus_keys_and_top_level(self):
        q = self.need("parse_nvidia_smi_xml")(NVQ_XML)
        self.assertEqual(set(q.get("gpus") or {}), {"0000:12:00.0", "0000:14:00.0"})
        self.assertEqual(q.get("driver_version"), "580.173.02")
        self.assertEqual(q.get("cuda_version"), "13.0")

    def test_p100_fields(self):
        self.need("parse_nvidia_smi_xml")
        g = _gpu(self, "0000:12:00.0")
        expect = {
            "product": "Tesla P100-PCIE-16GB", "brand": "Tesla", "arch": "Pascal",
            "serial": "0000000000001", "uuid": "GPU-00000000-0000-0000-0000-000000000001",
            "vbios": "86.00.52.00.02", "board_part": "900-2H400-6300-031",
            "gpu_part": "15F8-892-A1", "inforom_img": "H400.0201.00.08",
            "persistence": "Enabled", "display_active": "Disabled", "display_attached": "No",
            "compute_mode": "Default", "ecc_mode": "Enabled", "ecc_pending": "Enabled",
            "accounting": "Disabled", "virt": "None",
            "nv_gen_max": 2, "nv_gen_cur": 2, "nv_gen_dev_cur": 2, "nv_gen_dev_max": 3,
            "nv_gen_host_max": 2, "nv_width_max": 16, "nv_width_cur": 16,
            "pcie_replay": 0, "pcie_replay_rollover": 0, "pcie_tx_kbs": 0, "pcie_rx_kbs": 0,
            "pstate": "P0", "clock_reasons_active": ["gpu_idle"],
            "clock_reason_us": {"sw_power_cap": 427599, "sync_boost": 0,
                                "hw_therm_slowdown": 0, "hw_power_brake": 0},
            "vram_reserved": 115, "vram_free": 16265, "bar1_used": 2, "bar1_total": 16384,
            "enc_util": 0, "dec_util": 0, "enc_sessions": 0, "fbc_sessions": 0,
            "ecc_vol_sbe": 0, "ecc_vol_dbe": 0, "ecc_agg_sbe": 0, "ecc_agg_dbe": 0,
            "retired_sbe": 0, "retired_dbe": 0, "retired_pending": "No",
            "temp_shutdown": 85, "temp_slowdown": 82,
            "power_instant": 25.53, "power_enforced": 250.0, "power_requested": 250.0,
            "power_default": 250.0, "power_min": 125.0, "power_max": 250.0,
            "clk_video": 835, "clk_app_gfx": 1189, "clk_app_mem": 715,
            "clk_max_gfx": 1328, "clk_max_mem": 715, "clk_max_video": 1328,
        }
        for k, v in expect.items():
            self.assertEqual(g.get(k), v, k)
        for k in ("fan_pct", "gsp_fw", "temp_max_op", "temp_target", "temp_mem", "temp_tlimit",
                  "temp_mem_max", "jpeg_util", "ofa_util", "remapped_rows", "power_average"):
            self.assertNotIn(k, g)
        # the counter reported N/A is absent, not zero
        self.assertNotIn("sw_therm_slowdown", g.get("clock_reason_us") or {})

    def test_2060_fields(self):
        self.need("parse_nvidia_smi_xml")
        g = _gpu(self, "0000:14:00.0")
        expect = {
            "product": "NVIDIA GeForce RTX 2060", "brand": "GeForce", "arch": "Turing",
            "serial": "0000000000002", "gsp_fw": "580.173.02", "gpu_part": "1F08-200-A1",
            "display_active": "Enabled", "display_attached": "Yes",
            "fan_pct": 0, "pstate": "P8", "clock_reasons_active": ["sw_power_cap"],
            "clock_reason_us": {"sw_power_cap": 4991356120, "sync_boost": 0,
                                "sw_therm_slowdown": 0, "hw_therm_slowdown": 0,
                                "hw_power_brake": 0},
            "nv_gen_cur": 1, "nv_gen_dev_max": 3, "nv_gen_host_max": 2, "nv_width_cur": 4,
            "nv_width_max": 16, "pcie_tx_kbs": 683, "pcie_rx_kbs": 439,
            "vram_reserved": 409, "vram_free": 5566, "bar1_used": 14, "bar1_total": 256,
            "jpeg_util": 0, "ofa_util": 0,
            "temp_max_op": 88, "temp_target": 83, "temp_slowdown": 90, "temp_shutdown": 93,
            "power_instant": 2.06, "power_enforced": 190.0, "power_max": 240.0,
            "power_min": 125.0, "clk_max_gfx": 2175, "clk_max_mem": 7071, "clk_max_video": 2025,
        }
        for k, v in expect.items():
            self.assertEqual(g.get(k), v, k)
        for k in ("ecc_mode", "ecc_pending", "ecc_vol_sbe", "ecc_agg_dbe", "retired_sbe",
                  "retired_pending", "board_part", "clk_app_gfx", "clk_app_mem", "temp_mem"):
            self.assertNotIn(k, g)

    def test_old_tag_names_parse_to_the_same_values(self):
        old = (NVQ_XML.replace("<clocks_event_reasons>", "<clocks_throttle_reasons>")
               .replace("</clocks_event_reasons>", "</clocks_throttle_reasons>")
               .replace("clocks_event_reason_", "clocks_throttle_reason_")
               .replace("gpu_power_readings>", "power_readings>"))
        self.assertNotEqual(old, NVQ_XML)
        self.assertNotIn("<gpu_power_readings>", old)
        self.assertNotIn("<clocks_event_reason_sw_power_cap>", old)
        fn = self.need("parse_nvidia_smi_xml")
        self.assertEqual(fn(old).get("gpus"), fn(NVQ_XML).get("gpus"))

    def test_placeholder_values_are_skipped(self):
        fn = self.need("parse_nvidia_smi_xml")
        base = fn(NVQ_XML)["gpus"]["0000:14:00.0"]
        self.assertEqual(base.get("gsp_fw"), "580.173.02")
        for ph in ("[N/A]", "Not Supported", "[Not Supported]", ""):
            x = NVQ_XML.replace("<gsp_firmware_version>580.173.02<", f"<gsp_firmware_version>{ph}<")
            self.assertNotIn("gsp_fw", fn(x)["gpus"]["0000:14:00.0"], repr(ph))

    def test_garbage_gives_empty(self):
        fn = self.need("parse_nvidia_smi_xml")
        self.assertEqual(fn("")["gpus"], {})
        self.assertEqual(fn("<nvidia_smi_log><broken")["gpus"], {})


class ClockReasons(_Base):
    def test_classification(self):
        fn = self.need("nv_classify_clock_reasons")
        self.assertAlmostEqual(getattr(L, "_NV_POWER_CAP_NEAR", None) or 0, 0.80)
        # the measured case: 2060 in P8 at ~2 W of 190 W reports sw_power_cap Active
        self.assertEqual(fn(["sw_power_cap"], 2.06, 190.0), ([], ["sw power cap"]))
        self.assertEqual(fn(["sw_power_cap"], 240.0, 250.0), (["sw power cap"], []))
        self.assertEqual(fn(["sw_power_cap"], 200.0, 250.0), (["sw power cap"], []))   # at 80 %
        self.assertEqual(fn(["sw_power_cap"], 199.0, 250.0), ([], ["sw power cap"]))
        self.assertEqual(fn(["sw_power_cap"], None, 190.0), ([], ["sw power cap"]))
        self.assertEqual(fn(["sw_power_cap"], 100.0, None), ([], ["sw power cap"]))
        self.assertEqual(fn(["hw_thermal_slowdown"], 10.0, 250.0), (["hw thermal"], []))
        self.assertEqual(fn(["hw_slowdown", "hw_power_brake_slowdown", "sw_thermal_slowdown"],
                            1.0, 250.0), (["hw slowdown", "hw power brake", "sw thermal"], []))
        self.assertEqual(fn(["gpu_idle"], 25.0, 250.0), ([], ["idle"]))
        self.assertEqual(fn(["applications_clocks_setting", "display_clocks_setting",
                             "sync_boost"], 25.0, 250.0),
                         ([], ["app clock setting", "display clock setting", "sync boost"]))
        self.assertEqual(fn(["something_new"], 1.0, 250.0), ([], ["something_new"]))
        self.assertEqual(fn([], 1.0, 250.0), ([], []))


class NvidiaCorrelation(_Base):
    def test_bus_normalisation(self):
        fn = self.need("_nv_bus")
        self.assertEqual(fn("00000000:1B:00.0"), "0000:1b:00.0")
        self.assertEqual(fn("0000:1b:00.0"), "0000:1b:00.0")
        self.assertEqual(fn(" 00000000:12:00.0\n"), "0000:12:00.0")

    def test_index_resolved_by_bus_id(self):
        feed = _FakeFeed({0: _core(11.0), 2: _core(22.0)}, {"0000:1b:00.0": 0})
        g = L.NvidiaGpu("x", 2, feed)
        g.pci_addr = "0000:1b:00.0"
        self.assertEqual(g.sample().get("temp_main"), 11.0)

    def test_index_falls_back_to_enumeration_order(self):
        feed = _FakeFeed({0: _core(11.0), 2: _core(22.0)}, {})
        g = L.NvidiaGpu("x", 2, feed)
        g.pci_addr = "0000:1b:00.0"
        self.assertEqual(g.sample().get("temp_main"), 22.0)
        g2 = L.NvidiaGpu("x", 2, feed)               # no pci_addr at all
        self.assertEqual(g2.sample().get("temp_main"), 22.0)

    def test_feed_has_detail_state(self):
        f = L._NvidiaFeed()
        for a in ("detail_by_bus", "driver", "index_by_bus"):
            self.assertEqual(getattr(f, a, None), {}, f"behaviour missing: _NvidiaFeed.{a}")
        self.assertEqual(getattr(f, "detail_ts", None), 0.0)
        self.assertIs(f.stop, False)

    def test_dev_gives_pcie_and_detail_merges(self):
        self.assertIn("dev", inspect.signature(L.NvidiaGpu.__init__).parameters,
                      "behaviour missing: NvidiaGpu(dev=)")
        d = pci_tree(self.tmp.name)
        detail = dict(_gpu(self, "0000:14:00.0"))
        detail["util"] = 99.0                     # a core key: must not be overwritten
        feed = _FakeFeed({1: _core(32.0)}, {"0000:14:00.0": 1}, {"0000:14:00.0": detail},
                         {"driver_version": "580.173.02", "cuda_version": "13.0"},
                         time.monotonic())
        g = L.NvidiaGpu("x", 1, feed, dev=d)
        g.pci_addr = "0000:14:00.0"
        s = g.sample()
        self.assertEqual(s.get("util"), 10.0)
        self.assertEqual(s.get("pstate"), "P8")
        # round 7c (R4): CRIT at the slowdown point (90), WARN from max operating (88)
        self.assertEqual(s.get("temp_crit"), {"gpu": (88, 90)})
        self.assertEqual(s.get("throttle"), [])
        self.assertEqual(s.get("clock_reasons"), ["sw power cap"])
        self.assertEqual(s.get("nv_driver"), "580.173.02")
        self.assertEqual(s.get("cuda"), "13.0")
        self.assertEqual((s.get("pcie") or {}).get("card_gts"), 2.5)
        self.assertEqual((s.get("pcie") or {}).get("gen"), 1)
        self.assertEqual(s.get("temp"), {"gpu": 32.0})
        # the P100: CRIT at the slowdown point (82), no max op so WARN is the fixed 80 (round
        # 7c, R4); idle is a neutral reason, no dev → no pcie
        feed.detail_by_bus = {"0000:12:00.0": _gpu(self, "0000:12:00.0")}
        feed.by_index[0] = _core(38.0)
        g2 = L.NvidiaGpu("y", 0, feed)
        g2.pci_addr = "0000:12:00.0"
        s2 = g2.sample()
        self.assertEqual(s2.get("temp_crit"), {"gpu": (0, 82)})
        self.assertEqual(s2.get("clock_reasons"), ["idle"])
        self.assertEqual(s2.get("throttle"), [])
        self.assertIn("pcie", s2)
        self.assertIsNone(s2.get("pcie"))
        # memory temperature joins the temperature dict when the card reports one
        feed.detail_by_bus["0000:12:00.0"] = dict(_gpu(self, "0000:12:00.0"), temp_mem=41)
        self.assertEqual(g2.sample().get("temp"), {"gpu": 38.0, "mem": 41})

    def test_capped_under_load_is_throttle(self):
        detail = dict(_gpu(self, "0000:14:00.0"))
        core = _core(70.0)
        core["power"] = 185.0
        feed = _FakeFeed({1: core}, {"0000:14:00.0": 1}, {"0000:14:00.0": detail}, {},
                         time.monotonic())
        g = L.NvidiaGpu("x", 1, feed)
        g.pci_addr = "0000:14:00.0"
        s = g.sample()
        self.assertEqual(s.get("throttle"), ["sw power cap"])
        self.assertEqual(s.get("clock_reasons"), [])

    def test_stale_detail_is_not_merged(self):
        feed = _FakeFeed({1: _core(32.0)}, {}, {"0000:14:00.0": _gpu(self, "0000:14:00.0")}, {},
                         time.monotonic() - 60)
        g = L.NvidiaGpu("x", 1, feed)
        g.pci_addr = "0000:14:00.0"
        s = g.sample()
        self.assertNotIn("pstate", s)
        self.assertNotIn("throttle", s)
        self.assertEqual(s.get("temp_main"), 32.0)

    def test_old_construction_still_works(self):
        feed = _FakeFeed({0: _core(11.0)}, {})
        g = L.NvidiaGpu("x", 0, feed)
        s = g.sample()
        self.assertEqual(s.get("temp_main"), 11.0)
        self.assertEqual(s.get("temp"), {"gpu": 11.0})


class NvidiaRendering(_Base):
    def _sample2060(self):
        s = {"vendor": "NVIDIA", "name": "NVIDIA GeForce RTX 2060", "util": 10.0,
             "mem_util": 5.0, "vram_used": 171.0, "vram_total": 6144.0, "temp_main": 32.0,
             "temp": {"gpu": 32.0}, "power": 2.06, "power_cap": 190.0, "sclk": 300.0,
             "mclk": 405.0, "fan_rpm": None}
        s.update(_gpu(self, "0000:14:00.0"))
        s["throttle"], s["clock_reasons"] = [], ["sw power cap"]
        s["temp_crit"] = {"gpu": (88, 93)}
        s["nv_driver"], s["cuda"] = "580.173.02", "13.0"
        s["pcie"] = L.pcie_chain(pci_tree(self.tmp.name))
        return s

    def test_detail_rows_and_no_false_throttle(self):
        self.need("_nv_detail_rows")
        rows = L._gpu_rows(self._sample2060(), 140)
        t = _text(rows)
        for needle in ("P8", "held by", "sw power cap", "slowdown 90", "shutdown 93",
                       "max op 88", "target 83", "bar1 14/256 MiB", "fan 0%", "gen1 x4 now",
                       "580.173.02", "Turing", "1h23m11s", "power 190 W", "max 240"):
            self.assertIn(needle, t)
        self.assertNotIn("⚠ throttling", t)
        self.assertNotIn("0000000000002", t)
        self.assertNotIn("GPU-00000000-0000-0000-0000-000000000002", t)
        self.assertNotIn("serial", t.lower())
        self.assertNotIn("uuid", t.lower())
        # the core query parses to floats, so the existing renderer prints "32.0°C"
        self.assertEqual(_attrs_of(rows, "32.0°C"), [L.OK])

    def test_throttle_row_when_capped_under_load(self):
        s = self._sample2060()
        s["power"], s["throttle"], s["clock_reasons"] = 185.0, ["sw power cap"], []
        self.assertIn("⚠ throttling: sw power cap", _text(L._gpu_rows(s, 140)))

    def test_ecc_group_colours(self):
        fn = self.need("_nv_detail_rows")
        s = {"vendor": "NVIDIA", "name": "P100"}
        s.update(_gpu(self, "0000:12:00.0"))
        s["clock_reasons"] = ["idle"]           # as NvidiaGpu.sample sets it after classifying
        rows = fn(s, 136)
        t = _text(rows)
        self.assertIn("ecc", t)
        self.assertIn("Enabled", t)
        self.assertIn("held by: idle", t)
        self.assertNoWarnCrit(rows)
        s["ecc_vol_sbe"] = 3
        self.assertIn(L.WARN, _attrs_of(fn(s, 136), "sbe 3"))
        self.assertNotIn(L.CRIT, [a for _, a in _flat(fn(s, 136))])
        s["ecc_vol_dbe"] = 1
        self.assertIn(L.CRIT, _attrs_of(fn(s, 136), "dbe 1"))
        s["ecc_vol_sbe"], s["ecc_vol_dbe"], s["retired_pending"] = 0, 0, "Yes"
        self.assertIn(L.CRIT, _attrs_of(fn(s, 136), "Yes"))
        # a card without ECC gets no ecc group at all
        s2 = {"vendor": "NVIDIA", "name": "2060"}
        s2.update(_gpu(self, "0000:14:00.0"))
        self.assertNotIn("ecc", _text(fn(s2, 136)))

    def test_no_detail_no_rows(self):
        fn = self.need("_nv_detail_rows")
        self.assertEqual(fn({"vendor": "NVIDIA", "name": "x", "util": 1.0}, 100), [])
        # AMD publishes vram_free too; it is not an nvidia-smi detail
        self.assertEqual(fn({"vendor": "AMD", "name": "x", "vram_free": 100}, 100), [])

    def test_time_limited_counter_omits_zero_reasons(self):
        fn = self.need("_nv_detail_rows")
        s = {"vendor": "NVIDIA", "name": "x", "pstate": "P0",
             "clock_reason_us": {"sw_power_cap": 5_000_000, "sync_boost": 0}}
        t = _text(fn(s, 120))
        self.assertIn("sw power cap 5s", t)
        self.assertNotIn("sync boost", t)


# ================================================================ regression
class Regression(_Base):
    AMD = {"vendor": "AMD", "name": "Navi 22 [Radeon RX 6750 XT]",
           "temp_crit": {"edge": (100, 105), "junction": (110, 115), "mem": (100, 105)},
           "vram_used": 3000, "vram_total": 12272, "gtt_used": 1400, "gtt_total": 7900,
           "vram_free": 9272, "util": 42, "mem_util": 10, "sclk": 2245, "mclk": 675,
           "fclk": 1500, "socclk": 800,
           "temp": {"edge": 50, "junction": 55, "mem": 60, "vrm gfx": 47, "vrm soc": 45},
           "temp_main": 55, "power": 80.5, "power_cap": 230, "voltage": 900,
           "fan_rpm": 0, "fan_pct": 30, "metrics_version": "1.3", "mem_clock": 675,
           "gfx_clock": 2245, "vrm_temp_gfx": 47, "vrm_temp_soc": 45, "media_activity": 0,
           "throttle": ["board power (PPT0)"],
           "pcie": {"gts": 16.0, "width": 16, "gbs": 31.5, "narrow_gbs": 31.5,
                    "bottleneck": False, "narrow_gts": 16.0, "narrow_width": 16},
           "backend_dev": "Vulkan0", "pci_addr": "0000:03:00.0"}
    INTEL = {"vendor": "Intel", "name": "DG2 [Arc A770]", "util": 12.0,
             "engines": {"RCS": 12.0, "BCS": 0.0}, "sclk": 1350, "freq_min": 300,
             "freq_eff": 600, "freq_max": 2400, "freq_req": 1350, "throttle": [],
             "suspended": False, "temp_main": 45, "temp": {"gpu": 45}, "fan_rpm": None,
             "fan_pct": None, "power": None, "power_cap": None, "vram_used": 1000,
             "vram_total": 16384,
             "pcie": {"gts": 8.0, "width": 16, "gbs": 15.8, "narrow_gbs": 15.8,
                      "bottleneck": False, "narrow_gts": 8.0, "narrow_width": 16},
             "needs": {"power": "no energy counter (energy1_input) on this card"},
             "backend_dev": "Vulkan1", "pci_addr": "0000:16:00.0"}
    NV = {"vendor": "NVIDIA", "name": "Tesla P100-PCIE-16GB", "util": 0.0, "mem_util": 0.0,
          "vram_used": 5.0, "vram_total": 16384.0, "temp_main": 38.0, "power": 25.53,
          "power_cap": 250.0, "sclk": 405.0, "mclk": 715.0, "temp": {"gpu": 38.0},
          "backend_dev": None, "pci_addr": "0000:12:00.0"}
    NV_NEEDS = {"vendor": "NVIDIA", "name": "x",
                "needs": {f: "needs nvidia-smi" for f in ("util", "vram", "temp", "power", "clocks")}}

    @needs_baseline
    def test_gpu_rows_identical_for_old_samples(self):
        for s in (self.AMD, self.INTEL, self.NV, self.NV_NEEDS):
            for w in (70, 100, 140):
                self.assertEqual(L._gpu_rows(dict(s), w, 1), B._gpu_rows(dict(s), w, 1),
                                 f"{s['vendor']} at {w}")

    def test_fan_pct_without_tachometer(self):
        s = dict(self.NV, fan_rpm=None, fan_pct=0)
        rows = L._gpu_rows(s, 120)
        self.assertIn("fan 0%", _text(rows))
        self.assertEqual(sum(1 for t, _ in _flat(rows) if t == "fan "), 1)

    def test_fan_with_tachometer_draws_one_cell(self):
        s = dict(self.AMD, fan_rpm=1200, fan_pct=40)
        rows = L._gpu_rows(s, 120)
        self.assertIn("fan 1200 rpm (40%)", _text(rows))
        self.assertEqual(sum(1 for t, _ in _flat(rows) if t == "fan "), 1)

    @needs_baseline
    def test_fan_with_tachometer_unchanged(self):
        s = dict(self.AMD, fan_rpm=1200, fan_pct=40)
        rows = L._gpu_rows(s, 120)
        self.assertEqual(rows, B._gpu_rows(s, 120))

    @needs_baseline
    def test_nvsmi_core_query_unchanged(self):
        self.assertEqual(L._NVSMI_FIELDS, B._NVSMI_FIELDS)

    @needs_baseline
    def test_pcie_text_unchanged(self):
        link = dict(self.AMD["pcie"], card_gts=16.0, card_width=16, gen=4)
        self.assertEqual(L.pcie_text(link), B.pcie_text(link))
        self.assertEqual(L.pcie_text(None), B.pcie_text(None))


# ================================================================ D. hwmon sensors
class HwmonSensors(_Base):
    def _root(self, sub="hwmon", it87_temp1=38000, extra=None):
        root = os.path.join(self.tmp.name, sub)
        it87 = {"name": "it8728", "temp1_input": it87_temp1, "temp1_label": "CPU",
                "temp1_max": 60000, "temp1_crit": 70000, "temp2_input": -128000,
                "fan1_input": 1200, "in0_input": 1344, "pwm1": 128, "pwm1_enable": 1}
        it87.update(extra or {})
        ucsi = {"name": "ucsi_source_psy_0_00081", "curr1_input": 0, "curr1_max": 0,
                "in0_input": 0, "in0_max": 0, "in0_min": 0}
        k10 = {"name": "k10temp", "temp1_input": 40000, "temp1_max": 70000}
        fam = {"name": "fam15h_power", "power1_input": 5000000, "power1_crit": 125000000}
        for i, files in enumerate((it87, ucsi, k10, fam)):
            for k, v in files.items():
                _w(os.path.join(root, f"hwmon{i}"), k, v)
        return root, os.path.join(root, "hwmon2")

    def test_discovery_skips_excluded_chip_and_the_power_estimate(self):
        root, k10 = self._root()
        self.assertIn("fam15h_power", getattr(L, "_HWMON_SKIP", ()))
        chips = self.need("discover_hwmon_extra")(exclude=(k10,), root=root)
        self.assertEqual([c["name"] for c in chips], ["it8728", "ucsi_source_psy_0_00081"])
        # without the exclusion the CPU sensor is a chip like any other
        names = [c["name"] for c in self.need("discover_hwmon_extra")(root=root)]
        self.assertEqual(names, ["it8728", "ucsi_source_psy_0_00081", "k10temp"])

    def test_channels(self):
        root, k10 = self._root()
        chips = self.need("discover_hwmon_extra")(exclude=(k10,), root=root)
        it, ucsi = chips
        ch = {c["label"]: c for c in it["channels"]}
        self.assertEqual(ch["CPU"]["value"], 38)
        self.assertEqual(ch["CPU"]["limits"], (60, 70))
        self.assertIsNone(ch["temp2"]["value"])          # -128 °C: no diode, dropped by sane()
        self.assertEqual(ch["fan1"]["value"], 1200)
        self.assertAlmostEqual(ch["in0"]["value"], 1.344)
        self.assertEqual(ch["pwm1"]["value"], 50)
        self.assertNotIn("pwm1_enable", ch)
        self.assertEqual(it["device"], "virtual")
        u = {c["label"]: c for c in ucsi["channels"]}
        self.assertEqual(u["in0"]["value"], 0.0)          # 0 V is a reading on a USB-C source
        self.assertEqual(u["curr1"]["value"], 0.0)

    def test_device_is_the_basename_of_the_device_link(self):
        d = os.path.join(self.tmp.name, "hwmon", "hwmon5")
        _w(d, "name", "amdgpu")
        _w(d, "temp1_input", 50000)
        target = os.path.join(self.tmp.name, "pci0000:00", "0000:03:00.0")
        os.makedirs(target)
        os.symlink(target, os.path.join(d, "device"))
        chip = self.need("hwmon_chip")(d)
        self.assertEqual(chip["device"], "0000:03:00.0")
        self.assertEqual(chip["name"], "amdgpu")

    def test_chip_without_readable_channel_is_none(self):
        d = os.path.join(self.tmp.name, "hwmon", "hwmon7")
        _w(d, "name", "empty")
        self.assertIsNone(self.need("hwmon_chip")(d))
        self.assertEqual(self.need("discover_hwmon_extra")(root=os.path.dirname(d)), [])

    def test_rows(self):
        root, k10 = self._root()
        chips = self.need("discover_hwmon_extra")(exclude=(k10,), root=root)
        rows = self.need("_sensor_rows")(chips, 120)
        t = _text(rows)
        for needle in ("it8728", "1200 rpm", "CPU 38°C", "in0 1.344 V", "pwm1 50%", "in0 0.000 V",
                       "curr1 0.000 A", "virtual"):
            self.assertIn(needle, t)
        self.assertNotIn("temp2", t)
        self.assertEqual(_attrs_of(rows, "38°C"), [L.OK])
        self.assertEqual(_attrs_of(rows, "it8728"), [L.SER])
        self.assertNoWarnCrit(rows)

    def test_temperature_above_max_warns(self):
        root, k10 = self._root(sub="a", it87_temp1=65000)
        rows = self.need("_sensor_rows")(self.need("discover_hwmon_extra")(exclude=(k10,), root=root), 120)
        self.assertEqual(_attrs_of(rows, "65°C"), [L.WARN])
        root, k10 = self._root(sub="b", it87_temp1=75000)
        rows = self.need("_sensor_rows")(self.need("discover_hwmon_extra")(exclude=(k10,), root=root), 120)
        self.assertEqual(_attrs_of(rows, "75°C"), [L.CRIT])

    def test_alarm_flag_warns(self):
        root, k10 = self._root(sub="a", extra={"fan1_alarm": 1})
        rows = self.need("_sensor_rows")(self.need("discover_hwmon_extra")(exclude=(k10,), root=root), 120)
        self.assertEqual(_attrs_of(rows, "1200 rpm"), [L.WARN])
        root, k10 = self._root(sub="b", extra={"fan1_alarm": 0})
        rows = self.need("_sensor_rows")(self.need("discover_hwmon_extra")(exclude=(k10,), root=root), 120)
        self.assertEqual(_attrs_of(rows, "1200 rpm"), [0])

    def test_exclusion_list_from_the_monitors(self):
        class G:
            hw = os.path.join(self.tmp.name, "hwmon", "hwmon9")
        os.makedirs(G.hw)
        ex = self.need("_hwmon_exclude")([G(), object()])
        self.assertIn(os.path.realpath(G.hw), ex)
        if L.HW_CPU:
            self.assertIn(os.path.realpath(L.HW_CPU), ex)


# ================================================================ API surface
class Api(_Base):
    def test_draw_and_text_line_accept_sensors(self):
        for fn in (L.draw, L._text_line):
            p = inspect.signature(fn).parameters
            self.assertIn("sensors", p, f"behaviour missing: {fn.__name__}(sensors=)")
            self.assertIsNone(p["sensors"].default)

    _CPU = {"name": "x", "ncpu": 8, "util": 12.5, "temp": 40, "temps": {}, "freq": 4000,
            "power": None, "rapl_present": False, "load": ["0.1", "0.2", "0.3"], "loadavg": "0.1"}
    _ARGS = ([Regression.NV, Regression.AMD], _CPU, {"free": 9000, "total": 16000}, [], {}, [])

    def test_text_line_ignores_sensors(self):
        args = self._ARGS
        self.assertIn("sensors", inspect.signature(L._text_line).parameters,
                      "behaviour missing: _text_line(sensors=)")
        # the timestamp is stripped: two calls may straddle a second boundary
        self.assertEqual(L._text_line(*args, None).split(" | ", 1)[1],
                         L._text_line(*args).split(" | ", 1)[1])

    @needs_baseline
    def test_text_line_unchanged(self):
        args = self._ARGS
        a = L._text_line(*args).split(" | ", 1)[1]
        b = B._text_line(*args).split(" | ", 1)[1]
        self.assertEqual(a, b)
        self.assertEqual(L._text_line(*args, None).split(" | ", 1)[1], b)




# ================================================================ batch 2: audit fixes
def pci_chain_tree(tmp, nodes, aer_on=None, aer=None):
    """A chain of PCI nodes from the root port down to the device, each with a `vendor`
    file and its own link attributes: [(address, vendor, (cur_speed, cur_width),
    (max_speed, max_width)), ...]. Returns the path of the LAST node. `aer_on` names the
    address that gets the `aer` files."""
    p = os.path.join(tmp, "pci0000:00")
    last = None
    for addr, vendor, cur, mx in nodes:
        p = os.path.join(p, addr)
        os.makedirs(p, exist_ok=True)
        _w(p, "vendor", vendor)
        if cur:
            _w(p, "current_link_speed", f"{cur[0]} GT/s PCIe")
            _w(p, "current_link_width", cur[1])
        if mx:
            _w(p, "max_link_speed", f"{mx[0]} GT/s PCIe")
            _w(p, "max_link_width", mx[1])
        if addr == aer_on:
            for name, content in (aer or {}).items():
                _w(p, name, content)
        last = p
    return last


class PcieCardNode(_Base):
    """F2: the node whose link goes into the slot. Rule 1, a device under the root port
    negotiates with it (this machine, verified). Rule 2, behind an on-card switch the
    climb by PCI vendor finds the upstream port (heuristic, not verified on real switched
    hardware)."""
    ARC = [("0000:00:01.0", "0x8086", (16.0, 16), (16.0, 16)),
           ("0000:03:00.0", "0x8086", (16.0, 16), (16.0, 16)),
           ("0000:04:00.0", "0x8086", (2.5, 1), (2.5, 1)),
           ("0000:05:00.0", "0x8086", (2.5, 1), (2.5, 1))]
    NAVI = [("0000:00:01.0", "0x1022", (8.0, 16), (16.0, 16)),
            ("0000:03:00.0", "0x1002", (8.0, 16), (16.0, 16)),
            ("0000:04:00.0", "0x1002", (16.0, 16), (16.0, 16)),
            ("0000:05:00.0", "0x1002", (16.0, 16), (16.0, 16))]
    PLX = [("0000:00:01.0", "0x1022", (8.0, 16), (8.0, 16)),
           ("0000:03:00.0", "0x10b5", (8.0, 16), (8.0, 16)),
           ("0000:04:00.0", "0x10b5", (8.0, 8), (8.0, 16)),
           ("0000:05:00.0", "0x10de", (8.0, 8), (8.0, 16))]
    RADEON_SAME_VENDOR_AS_ROOT = [("0000:00:02.0", "0x1002", (5.0, 16), (5.0, 16)),
                                  ("0000:12:00.0", "0x1002", (5.0, 16), (8.0, 16))]
    THIS_2060 = [("0000:00:04.0", "0x1002", (2.5, 4), (5.0, 4)),
                 ("0000:14:00.0", "0x10de", (2.5, 4), (8.0, 16))]

    def test_arc_like_climbs_to_the_upstream_port(self):
        gpu = pci_chain_tree(self.tmp.name, self.ARC)
        node, via = self.need("_pcie_card_node")(gpu)
        self.assertEqual(node, os.path.realpath(os.path.dirname(os.path.dirname(gpu))))
        self.assertTrue(via)
        n = L._pcie_negotiation(gpu)
        self.assertEqual((n.get("gen"), n.get("card_width"), n.get("card_max_gen"),
                          n.get("card_max_width"), n.get("slot_max_gen"), n.get("slot_max_width")),
                         (4, 16, 4, 16, 4, 16))
        self.assertTrue(n.get("via_switch"))
        for flag in ("width_degraded", "speed_below_both", "width_limited_by_slot",
                     "speed_limited_by_host"):
            self.assertFalse(n.get(flag), flag)
        rows = L._pcie_rows({"pcie": L.pcie_chain(gpu)}, 120)
        self.assertIn("via on-card switch", _text(rows))
        self.assertIn("gen4 x16 now", _text(rows))
        self.assertNotIn("gen1 x1", _text(rows))
        self.assertNoWarnCrit(rows)

    def test_navi_like_reports_the_upstream_link(self):
        gpu = pci_chain_tree(self.tmp.name, self.NAVI)
        node, via = self.need("_pcie_card_node")(gpu)
        self.assertTrue(via)
        self.assertTrue(node.endswith("0000:03:00.0"))
        n = L._pcie_negotiation(gpu)
        self.assertEqual((n.get("gen"), n.get("card_width")), (3, 16))
        self.assertTrue(n.get("speed_below_both"))
        self.assertTrue(n.get("via_switch"))
        self.assertFalse(n.get("width_degraded"))
        self.assertIn("gen3 x16 now", _text(L._pcie_rows({"pcie": L.pcie_chain(gpu)}, 120)))

    def test_board_plx_switch_is_not_climbed(self):
        gpu = pci_chain_tree(self.tmp.name, self.PLX)
        node, via = self.need("_pcie_card_node")(gpu)
        self.assertEqual(node, os.path.realpath(gpu))
        self.assertFalse(via)
        n = L._pcie_negotiation(gpu)
        self.assertEqual((n.get("card_width"), n.get("slot_max_width")), (8, 16))
        self.assertTrue(n.get("width_degraded"))
        self.assertFalse(n.get("via_switch"))
        self.assertNotIn("via on-card switch", _text(L._pcie_rows({"pcie": L.pcie_chain(gpu)}, 120)))

    def test_same_vendor_as_the_root_port_stops_at_the_device(self):
        gpu = pci_chain_tree(self.tmp.name, self.RADEON_SAME_VENDOR_AS_ROOT)
        node, via = self.need("_pcie_card_node")(gpu)
        self.assertEqual(node, os.path.realpath(gpu))
        self.assertFalse(via)
        n = L._pcie_negotiation(gpu)
        self.assertEqual((n.get("gen"), n.get("card_max_gen"), n.get("slot_max_gen")), (2, 3, 2))
        self.assertTrue(n.get("speed_limited_by_host"))
        self.assertFalse(n.get("via_switch"))

    def test_this_machine_is_unchanged_plus_via_switch_false(self):
        gpu = pci_chain_tree(self.tmp.name, self.THIS_2060)
        with_vendor = L._pcie_negotiation(gpu)
        without = L._pcie_negotiation(pci_tree(os.path.join(self.tmp.name, "b")))
        self.assertEqual(with_vendor, without)
        self.assertIn("via_switch", with_vendor, "behaviour missing: via_switch")
        self.assertFalse(with_vendor["via_switch"])

    @needs_baseline
    def test_this_machine_keeps_the_old_pcie_chain_keys(self):
        gpu = pci_chain_tree(self.tmp.name, self.THIS_2060)
        self.assertEqual({k: L.pcie_chain(gpu)[k] for k in B.pcie_chain(gpu)}, B.pcie_chain(gpu))

    def test_missing_vendor_file_stops_the_climb(self):
        gpu = pci_tree(self.tmp.name)                        # no vendor files at all
        node, via = self.need("_pcie_card_node")(gpu)
        self.assertEqual(node, os.path.realpath(gpu))
        self.assertFalse(via)

    def test_aer_is_read_from_the_card_node(self):
        cor = "RxErr 2\nTOTAL_ERR_COR 2"
        aer = dict(AER_ZERO, aer_dev_correctable=cor)
        gpu = pci_chain_tree(self.tmp.name, self.ARC, aer_on="0000:03:00.0", aer=aer)
        n = L._pcie_negotiation(gpu)
        self.assertEqual((n.get("aer") or {}).get("correctable"), 2)
        gpu2 = pci_chain_tree(os.path.join(self.tmp.name, "b"), self.ARC, aer_on="0000:05:00.0", aer=aer)
        self.assertIsNone(L._pcie_negotiation(gpu2).get("aer"))
        # direct attach: the device is the card, as before
        gpu3 = pci_chain_tree(os.path.join(self.tmp.name, "c"), self.THIS_2060, aer_on="0000:14:00.0", aer=aer)
        self.assertEqual((L._pcie_negotiation(gpu3).get("aer") or {}).get("correctable"), 2)


class PcieThroughputBound(_Base):
    """F1: the driver reports impossible tx/rx figures (209715151 KB/s on an 8 GB/s link,
    seen on both P100s and by nvidia-smi dmon); the link's best possible bandwidth is a
    physical bound and sane() applies it."""
    def _p100_tree(self, sub, bus):
        return pci_tree(os.path.join(self.tmp.name, sub), dev=bus, port="0000:00:02.0",
                        cur=("5.0 GT/s PCIe", 16), mx=("8.0 GT/s PCIe", 16),
                        port_cur=("5.0 GT/s PCIe", 16), port_max=("5.0 GT/s PCIe", 16))

    def _sample(self, bus, tx, rx, dev):
        detail = dict(_gpu(self, "0000:12:00.0"), pcie_tx_kbs=tx, pcie_rx_kbs=rx)
        feed = _FakeFeed({0: _core(38.0)}, {bus: 0}, {bus: detail}, {}, time.monotonic())
        g = L.NvidiaGpu("p100", 0, feed, dev=dev)
        g.pci_addr = bus
        return g.sample()

    def test_glitch_is_dropped_and_not_rendered(self):
        bus = "0000:a1:00.0"
        s = self._sample(bus, 209715151, 48, self._p100_tree("a", bus))
        self.assertIsNone(s.get("pcie_tx_kbs"))
        self.assertEqual(s.get("pcie_rx_kbs"), 48)
        t = _text(L._pcie_rows(s, 140))
        self.assertNotIn("209715151", t)
        # no tx/rx cell at all, and nothing on the row above what the link can carry in
        # ANY unit: with adaptive units a glitch drawn as "209.7 GB/s" would have slipped
        # past a check on the "KB/s" text; the link's own "8.0 GB/s" cell stays under it
        self.assertNotIn("tx ", t)
        for num, unit in re.findall(r"(\d+(?:\.\d+)?) ([KMG])B/s", t):
            self.assertLessEqual(float(num) * {"K": 1, "M": 1e3, "G": 1e6}[unit], 8_000_000, t)

    def test_plausible_value_is_kept(self):
        bus = "0000:a2:00.0"
        s = self._sample(bus, 2103906, 25341, self._p100_tree("a", bus))
        self.assertEqual((s.get("pcie_tx_kbs"), s.get("pcie_rx_kbs")), (2103906, 25341))
        self.assertIn("tx 2.1 GB/s rx 25.3 MB/s", _text(L._pcie_rows(s, 140)))

    def test_zero_is_kept(self):
        bus = "0000:a3:00.0"
        s = self._sample(bus, 0, 0, self._p100_tree("a", bus))
        self.assertEqual((s.get("pcie_tx_kbs"), s.get("pcie_rx_kbs")), (0, 0))
        self.assertIn("tx 0 KB/s", _text(L._pcie_rows(s, 140)))

    def test_good_value_is_held_across_a_glitch(self):
        bus = "0000:a4:00.0"
        dev = self._p100_tree("a", bus)
        self.assertEqual(self._sample(bus, 25341, 7, dev).get("pcie_tx_kbs"), 25341)
        s = self._sample(bus, 209715151, 209715151, dev)
        self.assertEqual((s.get("pcie_tx_kbs"), s.get("pcie_rx_kbs")), (25341, 7))

    def test_at_the_bound_is_kept(self):
        bus = "0000:a5:00.0"                                  # 5.0 x16 slot: 8,000,000 KB/s
        s = self._sample(bus, 8000000, 8000001, self._p100_tree("a", bus))
        self.assertEqual(s.get("pcie_tx_kbs"), 8000000)
        self.assertIsNone(s.get("pcie_rx_kbs"))

    def test_without_negotiation_left_as_reported(self):
        bus = "0000:a6:00.0"
        s = self._sample(bus, 209715151, 48, None)
        self.assertEqual(s.get("pcie_tx_kbs"), 209715151)


class PcieRateUnits(_Base):
    """Round 8b: the tx/rx figures adapt their unit as they grow (KB/s, MB/s, GB/s, decimal
    like the link's GB/s on the same row and the bound), in the panel only; the sample,
    the bound and --probe keep the raw KB/s integers nvidia-smi reports."""
    def test_fmt_kbs_boundaries(self):
        fmt = self.need("_fmt_kbs")
        for kbs, text in ((0, "0 KB/s"), (999, "999 KB/s"), (1000, "1.0 MB/s"),
                          (25341, "25.3 MB/s"), (999999, "1000.0 MB/s"), (1000000, "1.0 GB/s"),
                          (2103906, "2.1 GB/s"), (8000000, "8.0 GB/s"), (None, "—")):
            with self.subTest(kbs=kbs):
                self.assertEqual(fmt(kbs), text)

    def _rows(self, tx, rx, columns):
        s = {"pcie": L.pcie_chain(pci_tree(self.tmp.name)), "pcie_tx_kbs": tx, "pcie_rx_kbs": rx}
        inner = columns - 2 - 4                              # box border, then _gpu_rows' margin
        rows = L._pcie_rows(s, inner)
        for r in rows:
            self.assertLessEqual(L._wwidth("".join(t for t, _ in r)), inner, _text(rows))
        return _text(rows)

    def test_adaptive_units_render_at_80_and_140_columns(self):
        for columns in (80, 140):
            with self.subTest(columns=columns):
                self.assertIn("tx 2.1 GB/s rx 25.3 MB/s", self._rows(2103906, 25341, columns))

    def test_fixture_values_keep_kbs_at_80_and_140_columns(self):
        # the captured values are all below 1000 and must read as they did (negative control)
        for bus, text in (("0000:14:00.0", "tx 683 KB/s rx 439 KB/s"),
                          ("0000:12:00.0", "tx 0 KB/s rx 0 KB/s")):
            g = _gpu(self, bus)
            for columns in (80, 140):
                with self.subTest(bus=bus, columns=columns):
                    self.assertIn(text, self._rows(g["pcie_tx_kbs"], g["pcie_rx_kbs"], columns))

    def test_probe_prints_the_raw_kbs(self):
        import contextlib
        import io
        detail = dict(_gpu(self, "0000:12:00.0"), pcie_tx_kbs=2103906, pcie_rx_kbs=25341)
        bus = "0000:a7:00.0"
        feed = _FakeFeed({0: _core(38.0)}, {bus: 0}, {bus: detail}, {}, time.monotonic())
        g = L.NvidiaGpu("p100", 0, feed, dev=None)
        g.pci_addr = bus
        stubs = {"discover_gpus": lambda: ([g], {}),
                 "cpu_sample": lambda: {"name": "cpu", "util": 0.0, "temp": None, "freq": None,
                                        "power": None, "rapl_present": False, "loadavg": "0",
                                        "temps": {}, "temp_limits": {}, "temp_relative": set()},
                 "mem_sample": lambda: {"free": 1, "total": 2, "swap_used": 0, "swap_total": 0},
                 "discover_hwmon_extra": lambda exclude=(): [],
                 "sample_llama_fleet": lambda port=None: []}
        for name, fn in stubs.items():
            real = getattr(L, name)
            setattr(L, name, fn)
            self.addCleanup(setattr, L, name, real)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            L.probe()
        text = buf.getvalue()
        self.assertIn("pcie_tx_kbs=2103906", text)
        self.assertIn("pcie_rx_kbs=25341", text)
        self.assertNotIn("GB/s", text)
        self.assertNotIn("MB/s", text)


class SmallRenderFixes(_Base):
    def test_sub_second_counter_reads_lt_1s(self):                     # F4
        s = {"vendor": "NVIDIA", "name": "x", "pstate": "P0",
             "clock_reason_us": {"sw_power_cap": 427599, "sync_boost": 0}}
        t = _text(L._nv_detail_rows(s, 120))
        self.assertIn("sw power cap <1s", t)
        self.assertNotIn("0s", t)
        self.assertNotIn("sync boost", t)
        s["clock_reason_us"] = {"sw_power_cap": 5_000_000}
        self.assertIn("sw power cap 5s", _text(L._nv_detail_rows(s, 120)))

    def test_chips_note_pluralises(self):                              # F5
        fn = self.need("_chips_note")
        self.assertEqual(fn(1), "1 chip")
        self.assertEqual(fn(2), "2 chips")
        self.assertIn("_chips_note(", inspect.getsource(L.draw))

    def test_speed_below_both_wording(self):                           # F6
        rows = L._pcie_rows({"pcie": L.pcie_chain(pci_tree(self.tmp.name))}, 120)
        self.assertIn("speed below both ends (normal at idle)", _text(rows))
        self.assertNotIn("reduced when idle", _text(rows))


class _FakeWin:
    """Records addnstr calls the way _put issues them."""
    def __init__(self, h, w):
        self.h, self.w, self.calls = h, w, []

    def getmaxyx(self):
        return self.h, self.w

    def addnstr(self, y, x, text, n, attr):
        self.calls.append((y, x, text, n, attr))


class BoxNote(_Base):                                                 # F7
    TITLE = "NVIDIA · GP100GL [Tesla P100 PCIe 16GB] (Vulkan1)"
    NOTE = "PCIe 5 GT/s x16 · 8.0 GB/s"

    FITTING = ((120, TITLE, NOTE), (60, "memory", ""), (40, "CPU · x", "8 threads"), (10, "t", "n"))

    def _calls(self, mod, width, title, note, cols=200):
        win = _FakeWin(40, cols)
        mod._box(win, 0, 1, width, 4, title, note)
        return win.calls

    def _both(self, width, title, note, cols=200):
        return self._calls(L, width, title, note, cols), self._calls(B, width, title, note, cols)

    def test_fitting_boxes_draw_their_note(self):
        for width, title, note in self.FITTING:
            a = self._calls(L, width, title, note)
            if note:
                self.assertTrue(any(note in t for _, _, t, _, _ in a))

    @needs_baseline
    def test_fitting_boxes_are_identical_to_the_backup(self):
        for width, title, note in self.FITTING:
            a, b = self._both(width, title, note)
            self.assertEqual(a, b, (width, title, note))

    def test_overflowing_note_is_omitted_whole(self):
        a = self._calls(L, 78, self.TITLE, self.NOTE, cols=80)
        # the note's own units; "PCIe" alone would also match the title "Tesla P100 PCIe 16GB"
        self.assertFalse([t for _, _, t, _, _ in a if "GB/s" in t or "GT/s" in t])
        self.assertTrue(any(self.TITLE in t for _, _, t, _, _ in a))
        top = [(x, t) for y, x, t, _, _ in a if y == 0]
        self.assertTrue(top[-1][1].endswith("─╮"))
        self.assertEqual(top[-1][0], 1 + 78 - 2)
        # the boundary: exactly fitting keeps the note, one column less drops it
        a = self._calls(L, 10, "t", "n")
        self.assertTrue([t for _, _, t, _, _ in a if " n " in t])
        a = self._calls(L, 9, "t", "n")
        self.assertFalse([t for _, _, t, _, _ in a if " n " in t])

    @needs_baseline
    def test_overflowing_note_differs_from_the_backup_only_when_it_overflows(self):
        a, b = self._both(78, self.TITLE, self.NOTE, cols=80)
        self.assertNotEqual(a, b)
        a, b = self._both(10, "t", "n")
        self.assertEqual(a, b)


# ================================================================ batch 3: AER honesty
class AerLive(_Base):
    """A: the counters only move when something can count them. Native AER needs the AER
    port service on the root port (<rootport>:pcie002 under /sys/bus/pci_express/devices);
    firmware-first needs an ACPI HEST table. This machine has neither (pcie_aspm=off keeps
    the kernel from requesting _OSC control, and there is no HEST), so its all-zero
    counters are a structural zero and the row must say why instead."""
    def _live_tree(self, root="0000:00:04.0", aer=AER_ZERO):
        return pci_tree(self.tmp.name, port=root, aer=aer)

    def test_not_live_row_holds_the_reason_and_no_figures(self):
        d = self._live_tree()
        _w(L._PCIE_SERVICES_DIR, "0000:00:04.0:pcie010", "")     # this machine's shape
        live, why = self.need("_aer_live")(d)
        self.assertFalse(live)
        self.assertIn("0000:00:04.0:pcie002 absent", why)
        self.assertIn("no ACPI HEST table", why)
        n = L._pcie_negotiation(d)
        self.assertFalse(n.get("aer_live"))
        self.assertEqual(n.get("aer_why"), why)
        rows = L._pcie_rows({"pcie": L.pcie_chain(d)}, 140)
        aer_rows = [r for r in rows if r[0][0].startswith("aer")]
        self.assertEqual(len(aer_rows), 1)
        t = "".join(x for x, _ in aer_rows[0])
        self.assertIn("not counted", t)
        self.assertNotIn("corrected", t)
        self.assertNotIn(" 0", t.replace("0000:00:04.0", ""))
        self.assertNoWarnCrit(rows)

    def test_live_via_native_service(self):
        d = self._live_tree(aer=dict(AER_ZERO, aer_dev_correctable="RxErr 3\nTOTAL_ERR_COR 3"))
        _w(L._PCIE_SERVICES_DIR, "0000:00:04.0:pcie002", "")
        live, why = self.need("_aer_live")(d)
        self.assertTrue(live)
        self.assertIsNone(why)
        rows = L._pcie_rows({"pcie": L.pcie_chain(d)}, 140)
        self.assertIn("corrected 3", _text(rows))
        self.assertNotIn("not counted", _text(rows))

    def test_live_via_firmware_first_only(self):
        d = self._live_tree()
        _w(L._PCIE_SERVICES_DIR, "0000:00:04.0:pcie010", "")
        _w(L._ACPI_TABLES_DIR, "HEST", "")
        live, why = self.need("_aer_live")(d)
        self.assertTrue(live)
        self.assertIn("corrected 0", _text(L._pcie_rows({"pcie": L.pcie_chain(d)}, 140)))

    def test_root_port_is_the_topmost_node_behind_a_switch(self):
        gpu = pci_chain_tree(self.tmp.name, PcieCardNode.ARC)
        _w(L._PCIE_SERVICES_DIR, "0000:00:01.0:pcie002", "")
        live, why = self.need("_aer_live")(os.path.dirname(os.path.dirname(gpu)))
        self.assertTrue(live, why)
        live, why = self.need("_aer_live")(gpu)
        self.assertTrue(live, why)
        os.remove(os.path.join(L._PCIE_SERVICES_DIR, "0000:00:01.0:pcie002"))
        live, why = self.need("_aer_live")(gpu)
        self.assertFalse(live)
        self.assertIn("root port 0000:00:01.0", why)

    def test_no_files_no_row_whatever_the_service(self):
        d = pci_tree(self.tmp.name)
        self.assertIsNone(L.pcie_aer(d))
        self.assertNotIn("aer", _text(L._pcie_rows({"pcie": L.pcie_chain(d)}, 140)))


class AerPort(_Base):
    """B: the link partner may count what the endpoint never sees (sysfs-bus-pci-devices-
    aer header), so the slot port's counters are read too, and drawn only while live."""
    def setUp(self):
        super().setUp()
        _w(L._PCIE_SERVICES_DIR, "0000:00:04.0:pcie002", "")

    def _tree(self, port_aer, card_aer=AER_ZERO, extra=None):
        d = pci_tree(self.tmp.name, aer=card_aer)
        port = os.path.dirname(d)
        for name, content in port_aer.items():
            _w(port, name, content)
        for name, content in (extra or {}).items():
            _w(port, name, content)
        return d, port

    def test_port_errors_during_the_session_warn_on_the_port_row_only(self):
        d, port = self._tree(AER_ZERO)
        n = L._pcie_negotiation(d)
        self.assertIsNotNone(n.get("aer_port"), "behaviour missing: aer_port")
        self.assertEqual(n["aer_port"]["correctable"], 0)
        _w(port, "aer_dev_correctable", "RxErr 4\nTOTAL_ERR_COR 4")
        rows = L._pcie_rows({"pcie": L.pcie_chain(d)}, 140)
        port_rows = [r for r in rows if r[0][0].startswith("port")]
        card_rows = [r for r in rows if r[0][0].startswith("aer")]
        self.assertEqual(len(port_rows), 1)
        self.assertIn("corrected 4 (+4 this session)", "".join(t for t, _ in port_rows[0]))
        self.assertIn("RxErr 4", "".join(t for t, _ in port_rows[0]))
        self.assertIn(L.WARN, [a for _, a in port_rows[0]])
        self.assertNotIn(L.WARN, [a for r in card_rows for _, a in r])
        self.assertNotIn(L.CRIT, [a for _, a in _flat(rows)])

    def test_port_uncorrectable_is_crit(self):
        d, port = self._tree(dict(AER_ZERO, aer_dev_nonfatal="Undefined 0\nMalfTLP 1\nTOTAL_ERR_NONFATAL 1"))
        rows = L._pcie_rows({"pcie": L.pcie_chain(d)}, 140)
        port_rows = [r for r in rows if r[0][0].startswith("port")]
        self.assertEqual(len(port_rows), 1)
        self.assertIn(L.CRIT, [a for _, a in port_rows[0]])
        self.assertIn("MalfTLP 1", "".join(t for t, _ in port_rows[0]))

    def test_rootport_totals_rendered_dim(self):
        d, port = self._tree(AER_ZERO, extra={"aer_rootport_total_err_cor": 7,
                                              "aer_rootport_total_err_fatal": 0,
                                              "aer_rootport_total_err_nonfatal": 1})
        a = L.pcie_aer(port)
        self.assertEqual((a.get("rootport_cor"), a.get("rootport_fatal"), a.get("rootport_nonfatal")),
                         (7, 0, 1))
        self.assertNotIn("rootport_cor", L.pcie_aer(d))           # the card has no such files
        rows = L._pcie_rows({"pcie": L.pcie_chain(d)}, 160)
        segs = [(t, a) for r in rows if r[0][0].startswith("port") for t, a in r]
        hit = [(t, a) for t, a in segs if "received from subtree" in t]
        self.assertEqual(len(hit), 1)
        self.assertIn("cor 7 · nonfatal 1 · fatal 0", hit[0][0])
        self.assertEqual(hit[0][1], L.DIM)
        self.assertNoWarnCrit(rows)

    def test_port_without_files_no_port_row(self):
        d = pci_tree(self.tmp.name, aer=AER_ZERO)
        n = L._pcie_negotiation(d)
        self.assertIn("aer_port", n, "behaviour missing: aer_port")
        self.assertIsNone(n["aer_port"])
        rows = L._pcie_rows({"pcie": L.pcie_chain(d)}, 140)
        self.assertFalse([r for r in rows if r[0][0].startswith("port")])
        self.assertIn("corrected 0", _text(rows))

    def test_not_live_no_port_row(self):
        os.remove(os.path.join(L._PCIE_SERVICES_DIR, "0000:00:04.0:pcie002"))
        d, port = self._tree(dict(AER_ZERO, aer_dev_correctable="RxErr 4\nTOTAL_ERR_COR 4"))
        rows = L._pcie_rows({"pcie": L.pcie_chain(d)}, 140)
        self.assertFalse([r for r in rows if r[0][0].startswith("port")])
        self.assertIn("not counted", _text(rows))
        self.assertNoWarnCrit(rows)


class NvidiaReplay(_Base):
    """C: the GPU's own PCIe replay counter, the one link-error counter that can move on
    this machine, as a per-session alarm."""
    def _sample(self, bus, replay, rollover, dev):
        detail = dict(_gpu(self, "0000:12:00.0"), pcie_replay=replay, pcie_replay_rollover=rollover)
        feed = _FakeFeed({0: _core(38.0)}, {bus: 0}, {bus: detail}, {}, time.monotonic())
        g = L.NvidiaGpu("p100", 0, feed, dev=dev)
        g.pci_addr = bus
        return g.sample()

    def _dev(self, bus):
        return pci_tree(self.tmp.name, dev=bus)

    def test_first_read_baselines(self):
        bus = "0000:b1:00.0"
        s = self._sample(bus, 5, 0, self._dev(bus))
        self.assertEqual((s.get("pcie_replay_session"), s.get("pcie_replay_rollover_session")), (0, 0))
        rows = L._pcie_rows(s, 140)
        self.assertIn("replays 5", _text(rows))
        self.assertNotIn("this session", _text(rows))
        self.assertNoWarnCrit(rows)

    def test_replays_during_the_session_warn(self):
        bus = "0000:b2:00.0"
        dev = self._dev(bus)
        self._sample(bus, 5, 0, dev)
        s = self._sample(bus, 8, 0, dev)
        self.assertEqual(s.get("pcie_replay_session"), 3)
        rows = L._pcie_rows(s, 140)
        self.assertIn("replays 8 (+3 this session)", _text(rows))
        self.assertIn(L.WARN, _attrs_of(rows, "+3 this session"))

    def test_rollover_during_the_session_warns(self):
        bus = "0000:b3:00.0"
        dev = self._dev(bus)
        self._sample(bus, 5, 1, dev)
        s = self._sample(bus, 5, 2, dev)
        self.assertEqual(s.get("pcie_replay_rollover_session"), 1)
        rows = L._pcie_rows(s, 140)
        self.assertIn("rollovers 2 (+1 this session)", _text(rows))
        self.assertIn(L.WARN, _attrs_of(rows, "+1 this session"))

    def test_counter_reset_rebaselines(self):
        bus = "0000:b4:00.0"
        dev = self._dev(bus)
        self._sample(bus, 9, 0, dev)
        s = self._sample(bus, 2, 0, dev)
        self.assertEqual(s.get("pcie_replay_session"), 0)
        self.assertNoWarnCrit(L._pcie_rows(s, 140))
        self.assertEqual(self._sample(bus, 3, 0, dev).get("pcie_replay_session"), 1)

    def test_no_replay_keys_nothing_drawn(self):
        bus = "0000:b5:00.0"
        detail = {k: v for k, v in _gpu(self, "0000:12:00.0").items()
                  if k not in ("pcie_replay", "pcie_replay_rollover")}
        feed = _FakeFeed({0: _core(38.0)}, {bus: 0}, {bus: detail}, {}, time.monotonic())
        g = L.NvidiaGpu("p100", 0, feed, dev=self._dev(bus))
        g.pci_addr = bus
        s = g.sample()
        self.assertNotIn("pcie_replay_session", s)
        self.assertNotIn("replays", _text(L._pcie_rows(s, 140)))


# ================================================================ batch 4: wrap the AER reason
class AerReasonWrap(_Base):
    """T2: at 80 columns the "not counted" reason was clipped by _put ("(0000:00:02.…"),
    losing "and no ACPI HEST table". Every other long row wraps; this one must too."""
    def _sample(self):
        d = pci_tree(self.tmp.name, dev="0000:12:00.0", port="0000:00:02.0",
                     cur=("5.0 GT/s PCIe", 16), mx=("8.0 GT/s PCIe", 16),
                     port_cur=("5.0 GT/s PCIe", 16), port_max=("5.0 GT/s PCIe", 16),
                     aer=AER_ZERO)
        _w(L._PCIE_SERVICES_DIR, "0000:00:02.0:pcie010", "")     # this machine's shape
        s = {"vendor": "NVIDIA", "name": "GP100GL [Tesla P100 PCIe 16GB]", "util": 0.0,
             "pcie": L.pcie_chain(d)}
        self.assertFalse(s["pcie"]["aer_live"])
        return s

    def _aer_rows(self, rows):
        out, on = [], False
        for r in rows:
            first = r[0][0]
            if first.startswith("aer"):
                on = True
            elif not first.startswith(" " * 8) or first.strip():
                on = False
            if on:
                out.append(r)
        return out

    def test_reason_wraps_at_eighty_columns(self):
        s = self._sample()
        why = s["pcie"]["aer_why"]
        width = 80 - 2                                    # draw(): bw = W - 2
        inner = width - 4                                 # _gpu_rows(): inner = width - 4
        rows = L._gpu_rows(s, width)
        aer = self._aer_rows(rows)
        self.assertGreaterEqual(len(aer), 2, "behaviour missing: the reason is not wrapped")
        text = " ".join("".join(t for t, _ in r) for r in aer)
        for word in why.split():
            self.assertIn(word, text, word)
        self.assertNotIn("…", text)
        for r in aer:
            self.assertLessEqual(L._cellw(r), inner)
            self.assertEqual({a for t, a in r if t.strip()}, {L.DIM})
        self.assertEqual(aer[0][0][0], f"{'aer':<8}")
        for r in aer[1:]:
            self.assertEqual(r[0], (" " * 8, 0))

    def test_reason_is_one_row_at_two_hundred_columns(self):
        s = self._sample()
        rows = L._pcie_rows(s, 200 - 2 - 4)
        aer = [r for r in rows if r[0][0].startswith("aer")]
        self.assertEqual(len(aer), 1)
        self.assertEqual(aer[0], [(f"{'aer':<8}", L.DIM), (s["pcie"]["aer_why"], L.DIM)])
        self.assertFalse([r for r in rows if r[0] == (" " * 8, 0) and "HEST" in "".join(t for t, _ in r)])


# ================================================================ batch 5: details menu
GROUP_LABELS = ("pcie", "aer", "port", "clocks", "limits", "memory", "ecc", "video", "id")


def _labelled(rows, label):
    """The rows whose 8-cell label column carries `label`."""
    return [r for r in rows if r[0][0].startswith(label) and r[0][0].strip() == label]


class DetailsMenu(_Base):
    """A dropdown with checkboxes to show or hide each detail group added since batch 1,
    with All / Minimal presets. Default: everything on, so a run that never opens the
    menu draws exactly what it drew before."""
    def _full_2060(self):
        d = pci_tree(self.tmp.name, aer=AER_ZERO)
        _w(L._PCIE_SERVICES_DIR, "0000:00:04.0:pcie002", "")          # live, with a port
        for name, content in dict(AER_ZERO, aer_dev_correctable="RxErr 1\nTOTAL_ERR_COR 1").items():
            _w(os.path.dirname(d), name, content)
        s = {"vendor": "NVIDIA", "name": "NVIDIA GeForce RTX 2060", "util": 10.0,
             "mem_util": 5.0, "vram_used": 171.0, "vram_total": 6144.0, "temp_main": 32.0,
             "temp": {"gpu": 32.0}, "power": 2.06, "power_cap": 190.0, "sclk": 300.0,
             "mclk": 405.0, "fan_rpm": None}
        s.update(_gpu(self, "0000:14:00.0"))
        s["throttle"], s["clock_reasons"] = [], ["sw power cap"]
        s["temp_crit"] = {"gpu": (88, 93)}
        s["nv_driver"], s["cuda"] = "580.173.02", "13.0"
        s["pcie"] = L.pcie_chain(d)
        return s

    def _full_p100(self):
        s = {"vendor": "NVIDIA", "name": "Tesla P100-PCIE-16GB", "util": 0.0,
             "vram_used": 5.0, "vram_total": 16384.0, "temp_main": 38.0, "temp": {"gpu": 38.0},
             "power": 25.53, "power_cap": 250.0, "sclk": 405.0, "mclk": 715.0}
        s.update(_gpu(self, "0000:12:00.0"))
        s["throttle"], s["clock_reasons"] = [], ["idle"]
        return s

    def _first_added(self, rows):
        """Index of the first row belonging to an added group, len(rows) when none."""
        for i, r in enumerate(rows):
            if any(_labelled([r], lab) for lab in GROUP_LABELS):
                return i
        return len(rows)

    def test_default_is_everything_on(self):
        details = self.need("_details")
        self.assertEqual(tuple(details), self.need("_DETAIL_GROUPS"))
        self.assertTrue(all(details.values()))
        self.assertEqual(self.need("_DETAIL_GROUPS"),
                         ("pcie", "aer", "clocks", "limits", "memory", "ecc", "video", "id", "sensors"))
        rows = L._gpu_rows(self._full_2060(), 140)
        for lab in ("pcie", "aer", "port", "clocks", "limits", "memory", "video", "id"):
            self.assertTrue(_labelled(rows, lab), lab)
        self.assertTrue(_labelled(L._gpu_rows(self._full_p100(), 140), "ecc"))
        self.assertTrue(self.need("_show")("pcie"))
        self.assertTrue(self.need("_show")("something-unknown"))

    def test_each_group_off_hides_only_that_group(self):
        details = self.need("_details")
        s2060, s100 = self._full_2060(), self._full_p100()
        # what each group owns in the rendering; "sensors" lives in draw(), tested below
        owns = {"pcie": ("pcie",), "aer": ("aer", "port"), "clocks": ("clocks",),
                "limits": ("limits",), "memory": ("memory",), "ecc": ("ecc",),
                "video": ("video",), "id": ("id",)}
        for group, labels in owns.items():
            sample = s100 if group == "ecc" else s2060
            full = L._gpu_rows(sample, 140)
            details[group] = False
            rows = L._gpu_rows(sample, 140)
            details[group] = True
            for lab in labels:
                self.assertFalse(_labelled(rows, lab), f"{group} off still draws {lab}")
                self.assertTrue(_labelled(full, lab), f"fixture lacks {lab}")
            others = [lab for g, labs in owns.items() if g != group for lab in labs
                      if _labelled(full, lab)]
            for lab in others:
                self.assertEqual(_labelled(rows, lab), _labelled(full, lab),
                                 f"{group} off changed {lab}")
            self.assertLess(len(rows), len(full), group)

    def test_minimal_is_the_panel_before_the_details(self):
        details = self.need("_details")
        for sample in (self._full_2060(), self._full_p100()):
            full = L._gpu_rows(sample, 140)
            cut = self._first_added(full)
            self.assertLess(cut, len(full))
            for g in details:
                details[g] = False
            rows = L._gpu_rows(sample, 140)
            for g in details:
                details[g] = True
            for lab in GROUP_LABELS:
                self.assertFalse(_labelled(rows, lab), lab)
            self.assertEqual(rows, full[:cut])
        for g in details:
            details[g] = False
        self.assertIn("fan 0%", _text(L._gpu_rows(self._full_2060(), 140)))
        c = {"name": "x", "ncpu": 8, "util": 1.0, "temp": 30, "temps": {"Tctl": 30}, "freq": 4000,
             "power": None, "rapl_present": False, "load": ["0", "0", "0"], "loadavg": "0",
             "temp_limits": {"Tctl": (70, 80)}, "temp_relative": ["Tctl"]}
        self.assertIn("Tctl 30°C rel", _text(L._cpu_rows(c, 120)))

    @needs_baseline
    def test_old_samples_match_the_baseline_in_both_presets(self):
        details = self.need("_details")
        for s in (Regression.AMD, Regression.INTEL, Regression.NV, Regression.NV_NEEDS):
            base = B._gpu_rows(dict(s), 120, 1)
            self.assertEqual(L._gpu_rows(dict(s), 120, 1), base)
            for g in details:
                details[g] = False
            self.assertEqual(L._gpu_rows(dict(s), 120, 1), base)
            for g in details:
                details[g] = True

    def test_menu_items_and_lines(self):
        details, menu = self.need("_details"), self.need("_menu")
        items = self.need("_menu_items")()
        self.assertEqual([lab for lab, _ in items][:2], ["All", "Minimal"])
        self.assertEqual([lab for lab, _ in items][2:], [f"[x] {g}" for g in L._DETAIL_GROUPS])
        details["clocks"] = details["video"] = False
        menu["cursor"] = 4                                    # "[x] limits"
        lines = self.need("_menu_lines")()
        self.assertEqual(lines, ["  All", "  Minimal", "  [x] pcie", "  [x] aer", "> [ ] clocks",
                                 "  [x] limits", "  [x] memory", "  [x] ecc", "  [ ] video",
                                 "  [x] id", "  [x] sensors"])

    def test_menu_keys(self):
        import curses
        details, menu = self.need("_details"), self.need("_menu")
        key = self.need("_menu_key")
        menu["open"], menu["cursor"] = True, 0
        key(curses.KEY_UP)
        self.assertEqual(menu["cursor"], 0)                   # clamped at the top
        for _ in range(20):
            key(ord("j"))
        self.assertEqual(menu["cursor"], len(L._DETAIL_GROUPS) + 1)   # clamped at the end
        key(curses.KEY_DOWN)
        self.assertEqual(menu["cursor"], len(L._DETAIL_GROUPS) + 1)
        key(ord("k"))
        self.assertEqual(menu["cursor"], len(L._DETAIL_GROUPS))
        menu["cursor"] = 2                                    # pcie
        key(ord(" "))
        self.assertEqual([g for g, on in details.items() if not on], ["pcie"])
        key(ord(" "))
        self.assertTrue(all(details.values()))
        menu["cursor"] = 1                                    # Minimal
        key(10)
        self.assertFalse(any(details.values()))
        menu["cursor"] = 0                                    # All
        key(13)
        self.assertTrue(all(details.values()))
        menu["cursor"] = 1
        key(curses.KEY_ENTER)
        self.assertFalse(any(details.values()))
        snapshot = dict(details)
        for k in (ord("q"), ord("m"), 27):
            menu["open"] = True
            key(k)
            self.assertFalse(menu["open"], k)
            self.assertEqual(details, snapshot, k)
        menu["open"] = True
        key(ord("x"))                                         # an unknown key: nothing
        self.assertTrue(menu["open"])
        self.assertEqual(details, snapshot)

    def test_tui_and_draw_wiring(self):
        tui = inspect.getsource(L._tui)
        self.assertIn("_menu_key(", tui, "behaviour missing: _tui feeds keys to the menu")
        self.assertIn('ord("m")', tui)
        self.assertIn("k = -1", tui)
        draw = inspect.getsource(L.draw)
        self.assertIn('_show("sensors")', draw, "behaviour missing: draw consults _show")
        self.assertIn("m details", draw)
        self.assertIn("_draw_menu(", draw)
        self.assertIn("m details menu", inspect.getsource(L.main))


# ================================================================ H. AMD smu7 (FirePro W9170)
# The literal contents of /sys/bus/pci/devices/0000:23:00.0 on the capture machine (FirePro
# W9170, physically an S9170: Hawaii XT GL, 1002:67a0, amdgpu with the legacy powerplay smu7
# backend, kernel 6.18.26), captured read-only at idle with the memory clock forced to level 1
# for a benchmark and runtime PM set to "on". Every readable file of the device directory,
# its hwmon node and fw_version/ is here verbatim, empty files included: fan1_input and
# fan1_target are EMPTY because the tachometer read fails (-EINVAL, TACH_PERIOD 0 on a
# passive board), and pp_force_state is empty because no state is forced. gpu_metrics is
# absent because the driver does not create it below GC 9.1.0, and that absence is the rule
# the reader uses to recognise a legacy card. The binary pp_table is left out.
W9170_SYSFS = {
    'board_info': 'type : cem\n',
    'class': '0x038000\n',
    'current_link_speed': '5.0 GT/s PCIe\n',
    'current_link_width': '1\n',
    'device': '0x67a0\n',
    'drm/entries': 'card1\ncontrolD65\nrenderD128\n',
    'fw_version/ce_fw_version': '0x0000007a\n',
    'fw_version/mc_fw_version': '0x00c79110\n',
    'fw_version/me_fw_version': '0x000000bb\n',
    'fw_version/mec_fw_version': '0x000001a5\n',
    'fw_version/pfp_fw_version': '0x000000e5\n',
    'fw_version/rlc_fw_version': '0x00000011\n',
    'fw_version/sdma2_fw_version': '0x0000004c\n',
    'fw_version/sdma_fw_version': '0x0000004c\n',
    'fw_version/smc_fw_version': '0x000f1500\n',
    'fw_version/uvd_fw_version': '0x01400900\n',
    'fw_version/vce_fw_version': '0x320a0200\n',
    'gpu_busy_percent': '0\n',
    'hwmon/hwmon2/fan1_enable': '0\n',
    'hwmon/hwmon2/fan1_input': '',
    'hwmon/hwmon2/fan1_max': '5700\n',
    'hwmon/hwmon2/fan1_min': '900\n',
    'hwmon/hwmon2/fan1_target': '',
    'hwmon/hwmon2/freq1_input': '300000000\n',
    'hwmon/hwmon2/freq1_label': 'sclk\n',
    'hwmon/hwmon2/freq2_input': '1250000000\n',
    'hwmon/hwmon2/freq2_label': 'mclk\n',
    'hwmon/hwmon2/in0_input': '1062\n',
    'hwmon/hwmon2/in0_label': 'vddgfx\n',
    'hwmon/hwmon2/name': 'amdgpu\n',
    'hwmon/hwmon2/power1_average': '43153000\n',
    'hwmon/hwmon2/power1_cap': '168000000\n',
    'hwmon/hwmon2/power1_cap_default': '168000000\n',
    'hwmon/hwmon2/power1_cap_max': '168000000\n',
    'hwmon/hwmon2/power1_cap_min': '0\n',
    'hwmon/hwmon2/power1_label': 'PPT\n',
    'hwmon/hwmon2/pwm1': '255\n',
    'hwmon/hwmon2/pwm1_enable': '2\n',
    'hwmon/hwmon2/pwm1_max': '255\n',
    'hwmon/hwmon2/pwm1_min': '0\n',
    'hwmon/hwmon2/temp1_crit': '104000\n',
    'hwmon/hwmon2/temp1_crit_hyst': '-273150\n',
    'hwmon/hwmon2/temp1_input': '42000\n',
    'hwmon/hwmon2/temp1_label': 'edge\n',
    'max_link_speed': '8.0 GT/s PCIe\n',
    'max_link_width': '16\n',
    'mem_busy_percent': '0\n',
    'mem_info_gtt_total': '8371179520\n',
    'mem_info_gtt_used': '17989632\n',
    'mem_info_preempt_used': '0\n',
    'mem_info_vis_vram_total': '17179869184\n',
    'mem_info_vis_vram_used': '4239360\n',
    'mem_info_vram_total': '34359738368\n',
    'mem_info_vram_used': '4378624\n',
    'power/control': 'on\n',
    'power/runtime_active_time': '1681984\n',
    'power/runtime_status': 'active\n',
    'power/runtime_suspended_time': '20237\n',
    'power_dpm_force_performance_level': 'manual\n',
    'power_dpm_state': 'performance\n',
    'power_state': 'D0\n',
    'pp_cur_state': '1\n',
    'pp_dpm_mclk': '0: 150Mhz \n1: 1250Mhz *\n',
    'pp_dpm_pcie': '0: 5.0GT/s, x16 *\n1: 5.0GT/s, x16 \n2: 5.0GT/s, x16 \n3: 5.0GT/s, x16 \n4: 5.0GT/s, x16 \n5: 5.0GT/s, x16 \n',
    'pp_dpm_sclk': '0: 300Mhz *\n1: 461Mhz \n2: 618Mhz \n3: 774Mhz \n4: 824Mhz \n5: 861Mhz \n6: 900Mhz \n7: 930Mhz \n',
    'pp_force_state': '\n',
    'pp_mclk_od': '0\n',
    'pp_num_states': 'states: 2\n0 boot\n1 performance\n',
    'pp_power_profile_mode': 'NUM        MODE_NAME     SCLK_UP_HYST   SCLK_DOWN_HYST SCLK_ACTIVE_LEVEL     MCLK_UP_HYST   MCLK_DOWN_HYST MCLK_ACTIVE_LEVEL\n  0   BOOTUP_DEFAULT:        -                -                -                -                -                -\n  1 3D_FULL_SCREEN *:        0              100               30                0              100               10\n  2     POWER_SAVING:       10                0               30                -                -                -\n  3            VIDEO:        -                -                -               10               16               31\n  4               VR:        0               11               50                0              100               10\n  5          COMPUTE:        0                5               30                -                -                -\n  6           CUSTOM:        -                -                -                -                -                -\n',
    'pp_sclk_od': '0\n',
    'revision': '0x00\n',
    'subsystem_device': '0x0735\n',
    'subsystem_vendor': '0x1002\n',
    'thermal_throttling_logging': '0000:23:00.0: thermal throttling logging enabled, with interval 60 seconds\n',
    'vbios_build': '329701  \n',
    'vbios_version': '113-C6812000-100\n',
    'vendor': '0x1002\n',
}


def _group(rows, label):
    """Every row of one detail group: the row carrying `label` in its 8-cell label column
    and the continuation rows under it (blank label column), up to the next labelled row.
    _labelled() returns only the first row, which hides a cell that wrapped."""
    out, on = [], False
    for r in rows:
        head = r[0][0]
        if head.strip():
            on = head.strip() == label
        if on:
            out.append(r)
    return out


class _Touched(BaseException):
    """Raised by the open() tripwire below when the reader opens a file it must not. A
    BaseException on purpose: read() and read_int() swallow every Exception and answer
    None, which would turn a forbidden read into a silent dash and the test into a pass."""


def w9170_tree(tmp, files=None, gpu_metrics=False, busy=None):
    """<tmp>/pci0000:00/0000:00:0b.0/0000:23:00.0 holding the fixture files, and
    <tmp>/card1/device symlinked to it, the shape AmdGpu(card, name) reads (the real
    /sys/class/drm/cardN/device is such a link). The root port carries no link attributes,
    so nothing about the slot is invented: the card node alone says 5.0 GT/s x1 against its
    own 8.0 GT/s x16 maximum. `files` replaces the fixture wholesale, `busy` overrides the
    two activity files, `gpu_metrics=True` adds the file a swsmu/Navi card publishes, which
    turns the reader's legacy rule off."""
    files = dict(W9170_SYSFS if files is None else files)
    if busy is not None:
        files["gpu_busy_percent"] = f"{busy}\n"
        files["mem_busy_percent"] = f"{busy}\n"
    dev = os.path.join(tmp, "pci0000:00", "0000:00:0b.0", "0000:23:00.0")
    for rel, content in files.items():
        p = os.path.join(dev, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(content)                 # verbatim: an empty file stays empty
    if gpu_metrics:
        with open(os.path.join(dev, "gpu_metrics"), "wb") as f:
            f.write(b"")
    card = os.path.join(tmp, "card1")
    os.makedirs(card)
    os.symlink(dev, os.path.join(card, "device"))
    return card


class AmdSmu7(_Base):
    """Full amdgpu data for a legacy powerplay (smu7) card, read without ever sending an SMC
    message from the periodic sample while the GPU is busy. On 20/04 a loop reading
    pp_dpm_mclk during a Vulkan workload froze the capture machine
    (smu7_print_clock_levels -> ci_send_msg_to_smc -> phm_wait_for_register_unequal); the
    five files whose read sends such a message are the tripwire list below."""
    SMC = ("pp_dpm_sclk", "pp_dpm_mclk", "freq1_input", "freq2_input", "power1_average")

    def setUp(self):
        super().setUp()
        self._opened = []
        self._forbid = ()
        real_open = open

        def trip(path, *a, **k):
            base = os.path.basename(str(path))
            self._opened.append(base)
            if base in self._forbid:
                raise _Touched(base)
            return real_open(path, *a, **k)
        # a module-level name shadows the builtin inside read(), read_int() and the
        # gpu_metrics reader, so every sysfs open of the program under test passes here
        L.open = trip
        self.addCleanup(lambda: delattr(L, "open") if "open" in vars(L) else None)

    def _gpu(self, sub="", **kw):
        card = w9170_tree(os.path.join(self.tmp.name, sub) if sub else self.tmp.name, **kw)
        return L.AmdGpu(card, "FirePro W9170")

    def _sample(self, g):
        try:
            return g.sample()
        except _Touched as e:
            self.fail(f"the periodic sample opened {e}, an SMC-message file, on a busy smu7 card")

    def _amd(self, s):
        """The detail dict: fail (not error) with a readable message while it is missing."""
        self.assertIn("amd", s, "behaviour missing: AmdGpu.sample()['amd']")
        return s["amd"]

    def _set_busy(self, g, pct):
        dev = os.path.realpath(g.dev)
        for n in ("gpu_busy_percent", "mem_busy_percent"):
            with open(os.path.join(dev, n), "w") as f:
                f.write(f"{pct}\n")

    # ---- the rule that tells a legacy card apart, and the classification it rests on
    def test_legacy_rule_is_the_absence_of_gpu_metrics(self):
        g = self._gpu()
        self.assertTrue(getattr(g, "legacy", None), "behaviour missing: AmdGpu.legacy")
        navi = self._gpu("b", gpu_metrics=True)
        self.assertIs(navi.legacy, False)

    def test_smc_message_files_are_declared_by_the_program(self):
        self.assertEqual(set(self.need("_AMD_SMC_MESSAGE_FILES")), set(self.SMC))

    # ---- AC-2a positive: busy smu7 card, no message file is ever opened
    def test_busy_smu7_never_opens_an_smc_message_file(self):
        g = self._gpu(busy=100)
        self._forbid = self.SMC
        for _ in range(5):
            s = self._sample(g)
        # the register-only readings still arrive every tick
        self.assertEqual(s["util"], 100)
        self.assertEqual(s["mem_util"], 100)
        self.assertEqual(s["temp"]["edge"], 42)
        self.assertEqual(s["voltage"], 1062)
        self.assertEqual(s["fan_pct"], 100)
        self.assertEqual((s["vram_used"], s["vram_total"]), (4, 32768))
        self.assertIsNone(s.get("sclk"))
        self.assertIsNone(s.get("mclk"))
        self.assertIsNone(s.get("power"))
        self.assertIn("idle only", s["needs"]["clocks"])
        self.assertIn("idle only", s["needs"]["power"])

    def test_without_a_busy_signal_the_slow_tier_never_runs(self):
        files = {k: v for k, v in W9170_SYSFS.items() if k not in ("gpu_busy_percent", "mem_busy_percent")}
        g = self._gpu(files=files)
        self._forbid = self.SMC
        s = self._sample(g)
        self.assertIsNone(s["util"])
        self.assertIsNone(s.get("sclk"))
        self.assertIn("gpu_busy_percent", s["needs"]["clocks"])

    # ---- AC-2b negative: a card with gpu_metrics keeps the per-tick reads
    def test_navi_like_card_reads_clocks_every_tick(self):
        g = self._gpu(gpu_metrics=True, busy=100)
        self.assertIs(getattr(g, "legacy", None), False, "behaviour missing: AmdGpu.legacy")
        for _ in range(3):
            self._opened.clear()
            s = g.sample()
            self.assertIn("pp_dpm_sclk", self._opened)
            self.assertIn("pp_dpm_mclk", self._opened)
            self.assertIn("power1_average", self._opened)
        self.assertEqual((s["sclk"], s["mclk"]), (300, 1250))
        self.assertAlmostEqual(s["power"], 43.153, places=3)
        self.assertNotIn("idle only", str(s.get("needs", {})))
        self.assertIsNone(s.get("slow_age"))

    # ---- the slow tier: idle by the register-only signal, once per interval
    def test_idle_smu7_reads_the_slow_tier_once_per_interval(self):
        saved = self.need("_AMD_SLOW_INTERVAL")
        self.addCleanup(setattr, L, "_AMD_SLOW_INTERVAL", saved)
        g = self._gpu()                                  # the fixture is idle: 0 %
        self._opened.clear()
        s = g.sample()
        self.assertEqual([b for b in self._opened if b in self.SMC].count("pp_dpm_sclk"), 1)
        self.assertEqual((s["sclk"], s["mclk"]), (300, 1250))
        self.assertAlmostEqual(s["power"], 43.153, places=3)
        # round 7c (R1): the hwmon clocks are no longer read, so they are not in the sample
        self.assertNotIn("hw_sclk", s["amd"])
        self.assertNotIn("hw_mclk", s["amd"])
        self.assertEqual(set(s["slow_fields"]), {"sclk", "mclk", "power"})
        self.assertLess(s["slow_age"], 1.0)
        self.assertNotIn("clocks", s.get("needs", {}))
        self.assertNotIn("power", s.get("needs", {}))
        # a second tick inside the interval: no message, the values are kept and age
        self._forbid = self.SMC
        s2 = self._sample(g)
        self.assertEqual((s2["sclk"], s2["mclk"], s2["power"]), (s["sclk"], s["mclk"], s["power"]))
        self.assertGreaterEqual(s2["slow_age"], s["slow_age"])
        # after the interval: read again
        self._forbid = ()
        L._AMD_SLOW_INTERVAL = 0.0
        self._opened.clear()
        g.sample()
        self.assertIn("pp_dpm_sclk", self._opened)
        # the card gets busy: the idle values are NOT replayed as live
        L._AMD_SLOW_INTERVAL = saved
        self._set_busy(g, 100)
        self._forbid = self.SMC
        s3 = self._sample(g)
        self.assertIsNone(s3["sclk"])
        self.assertIsNone(s3["power"])
        self.assertIn("idle only", s3["needs"]["clocks"])
        self.assertIn("s ago", s3["needs"]["clocks"])
        self.assertEqual(s3["amd"]["sclk_levels"], [300, 461, 618, 774, 824, 861, 900, 930])
        self.assertIsNone(s3["amd"]["sclk_active"])

    def test_idle_threshold_both_sides(self):
        thr = self.need("_AMD_IDLE_BUSY")
        g = self._gpu(busy=thr)
        self._opened.clear()
        g.sample()
        self.assertIn("pp_dpm_sclk", self._opened, "at the threshold the card counts as idle")
        g2 = self._gpu("b", busy=thr + 1)
        self._forbid = self.SMC
        self._sample(g2)

    # ---- AC-2c / AC-2f: every field from the fixture, with the tables and active levels
    def test_w9170_fields_from_the_fixture(self):
        s = self._gpu().sample()
        self.assertIn("amd", s, "behaviour missing: AmdGpu.sample()['amd']")
        a = s["amd"]
        expected = {
            "vbios": "113-C6812000-100", "vbios_build": "329701", "board": "cem",
            "fw": {"ce": "0x0000007a", "mc": "0x00c79110", "mec": "0x000001a5",
                   "me": "0x000000bb", "pfp": "0x000000e5", "rlc": "0x00000011",
                   "sdma2": "0x0000004c", "sdma": "0x0000004c", "smc": "0x000f1500",
                   "uvd": "0x01400900", "vce": "0x320a0200"},
            "vis_vram_used": 4, "vis_vram_total": 16384, "preempt_used": 0,
            "perf_level": "manual", "dpm_state": "performance",
            "states": [(0, "boot"), (1, "performance")], "cur_state": 1, "forced_state": None,
            "profile": "3D_FULL_SCREEN", "sclk_od": 0, "mclk_od": 0,
            "pcie_levels": ["5.0GT/s, x16"] * 6, "pcie_active": 0,
            "sclk_levels": [300, 461, 618, 774, 824, 861, 900, 930], "sclk_active": 0,
            "mclk_levels": [150, 1250], "mclk_active": 1,
            "power_cap_default": 168, "power_cap_min": 0, "power_cap_max": 168,
            "power_label": "PPT", "temp_crit": 104, "temp_crit_hyst": None,
            "fan_min": 900, "fan_max": 5700, "fan_target": None, "fan_mode": "auto",
            "fan_enable": 0, "pwm_min": 0, "pwm_max": 255,
            "throttle_log": ("enabled", 60), "pci_power_state": "D0",
            "runtime_control": "on", "runtime_status": "active", "missing": [],
        }
        for k, v in expected.items():
            self.assertEqual(a.get(k), v, k)
        self.assertIn("no tachometer reading", a["fan_why"])
        self.assertEqual(s["power_cap"], 168)
        self.assertIsNone(s["fan_rpm"])
        self.assertEqual(s["fan_pct"], 100)
        self.assertEqual(s["pcie"]["card_width"], 1)
        self.assertTrue(s["pcie"]["width_degraded"])

    def test_dpm_table_parser(self):
        levels = self.need("_amd_dpm_levels")
        self.assertEqual(levels("0: 300Mhz *\n1: 461Mhz \n"), (["300Mhz", "461Mhz"], 0))
        self.assertEqual(levels(W9170_SYSFS["pp_dpm_pcie"]), (["5.0GT/s, x16"] * 6, 0))
        self.assertEqual(levels(W9170_SYSFS["pp_dpm_mclk"]), (["150Mhz", "1250Mhz"], 1))
        self.assertEqual(levels("0: 300Mhz \n1: 461Mhz \n"), (["300Mhz", "461Mhz"], None))
        self.assertEqual(levels(""), ([], None))
        self.assertEqual(levels(None), ([], None))

    def test_sane_bounds_on_the_new_numerics(self):
        bad = dict(W9170_SYSFS)
        # round 7c (R1): freq1_input/freq2_input are never read, so no bad value of theirs
        # can reach the sample; their bounds check is gone with the read
        bad.update({"hwmon/hwmon2/power1_cap_max": "999999999999999\n",
                    "hwmon/hwmon2/power1_cap_default": "-1\n",
                    "hwmon/hwmon2/fan1_max": "999999\n", "hwmon/hwmon2/temp1_crit": "500000\n",
                    "pp_sclk_od": "500\n", "pp_cur_state": "7\n",
                    "pp_dpm_sclk": "0: 0Mhz *\n1: 99999Mhz \n",
                    "mem_info_vis_vram_used": "99999999999999999\n",
                    "thermal_throttling_logging": "x: thermal throttling logging enabled, with interval 0 seconds\n"})
        a = self._amd(self._gpu(files=bad).sample())
        for k in ("power_cap_max", "power_cap_default", "fan_max",
                  "temp_crit", "sclk_od", "cur_state", "sclk_levels", "sclk_active",
                  "vis_vram_used", "throttle_log"):
            self.assertIsNone(a.get(k), k)
        self.assertEqual(a["power_cap_min"], 0)          # 0 is the driver's literal, kept

    # ---- AC-2d: absent files give None and a reason, never a zero
    def test_absent_files_give_none_and_a_reason(self):
        gone = ("vbios_version", "vbios_build", "board_info", "mem_info_vis_vram_used",
                "mem_info_vis_vram_total", "mem_info_preempt_used", "pp_power_profile_mode",
                "pp_sclk_od", "pp_mclk_od", "pp_num_states", "pp_cur_state", "pp_force_state",
                "pp_dpm_pcie", "power_dpm_state", "power_dpm_force_performance_level",
                "thermal_throttling_logging", "power_state", "power/control",
                "power/runtime_status", "hwmon/hwmon2/power1_cap_default",
                "hwmon/hwmon2/power1_cap_min", "hwmon/hwmon2/power1_cap_max",
                "hwmon/hwmon2/power1_label", "hwmon/hwmon2/fan1_min", "hwmon/hwmon2/fan1_max",
                "hwmon/hwmon2/fan1_target", "hwmon/hwmon2/pwm1_enable", "hwmon/hwmon2/fan1_enable",
                "hwmon/hwmon2/pwm1_min", "hwmon/hwmon2/pwm1_max", "hwmon/hwmon2/temp1_crit",
                "hwmon/hwmon2/temp1_crit_hyst")
        files = {k: v for k, v in W9170_SYSFS.items()
                 if k not in gone and not k.startswith("fw_version/")}
        a = self._amd(self._gpu(files=files).sample())
        for k in ("vbios", "vbios_build", "board", "fw", "vis_vram_used", "vis_vram_total",
                  "preempt_used", "profile", "sclk_od", "mclk_od", "states", "cur_state",
                  "forced_state", "pcie_levels", "pcie_active", "dpm_state", "perf_level",
                  "throttle_log", "pci_power_state", "runtime_control", "runtime_status",
                  "power_cap_default", "power_cap_min", "power_cap_max", "power_label",
                  "fan_min", "fan_max", "fan_target", "fan_mode", "fan_enable", "pwm_min",
                  "pwm_max", "temp_crit", "temp_crit_hyst"):
            self.assertIsNone(a.get(k), k)
        for name in gone:
            base = os.path.basename(name)
            self.assertTrue(any(m.endswith(base) for m in a["missing"]), name)
        self.assertIn("fw_version", a["missing"])

    # ---- AC-2e: the labels, in the panel
    def test_slow_values_are_labelled_in_the_panel(self):
        g = self._gpu()
        idle = g.sample()
        rows = L._gpu_rows(idle, 120)
        text = _text(rows)
        self.assertIn("core 300 MHz", text)
        self.assertIn("idle only", text)
        self.assertRegex(text, r"idle only[^│]*\ds ago")
        self.assertIn("idle only", _text(_group(rows, "clocks")))
        self._set_busy(g, 100)
        busy = g.sample()
        rows = L._gpu_rows(busy, 120)
        text = _text(rows)
        self.assertNotIn("core 300 MHz", text)
        self.assertNotIn("power 43", text)
        self.assertIn("no data", text)
        self.assertRegex(text, r"power, clocks — idle only.*\ds ago")
        self.assertIn("300 461 618", _text(_group(rows, "clocks")))   # the static list stays

    def test_reason_rows_fit_at_80_columns(self):
        # Written AFTER the fix, as a guard: the first-read reasons were 80 and 78 cells
        # on a 74-cell row and _put clipped them, found by rendering, not by a test.
        first = self._gpu(busy=100).sample()
        files = {k: v for k, v in W9170_SYSFS.items() if k not in ("gpu_busy_percent", "mem_busy_percent")}
        blind = self._gpu("b", files=files).sample()
        g = self._gpu("c")
        g.sample()
        self._set_busy(g, 100)
        busy = g.sample()
        for s in (first, blind, busy):
            rows = [r for r in L._gpu_rows(s, 78) if "idle only" in _text([r])]
            self.assertTrue(rows)
            for r in rows:
                self.assertLessEqual(L._wwidth("".join(t for t, _ in r)), 74, _text([r]))

    def test_static_identity_is_labelled_at_start(self):
        rows = L._gpu_rows(self._gpu().sample(), 200)
        self.assertIn("at start", _text(_group(rows, "id")))

    # ---- AC-3: the detail groups, at 80 and 200 columns
    def test_amd_detail_groups_at_80_and_200_columns(self):
        s = self._gpu().sample()
        for width in (78, 198):                       # draw() hands _gpu_rows W - 2
            rows = L._gpu_rows(s, width)
            for lab in ("pcie", "clocks", "limits", "memory", "id"):
                self.assertTrue(_labelled(rows, lab), f"{lab} at {width}")
            for r in rows:
                self.assertLessEqual(L._wwidth("".join(t for t, _ in r)), width - 4, width)
            self.assertFalse([t for t, a in _flat(rows) if a == L.CRIT], "CRIT on a healthy card")
        text = _text(L._gpu_rows(s, 198))
        for needle in ("vbios 113-C6812000-100", "build 329701", "board cem", "smc 0x000f1500",
                       "3D_FULL_SCREEN", "manual", "performance", "150 1250*", "300* 461",
                       "168 W", "104", "900–5700", "no tachometer reading", "16384 MiB",
                       "preempt 0", "D0", "runtime pm on", "throttling logging enabled"):
            self.assertIn(needle, text, needle)
        # each group off hides only its own rows, as for the NVIDIA groups
        details = self.need("_details")
        full = L._gpu_rows(s, 140)
        for group in ("clocks", "limits", "memory", "id"):
            details[group] = False
            rows = L._gpu_rows(s, 140)
            details[group] = True
            self.assertFalse(_labelled(rows, group), group)
            for other in ("pcie", "clocks", "limits", "memory", "id"):
                if other != group:
                    self.assertEqual(_labelled(rows, other), _labelled(full, other), other)

    def test_old_amd_sample_draws_no_amd_detail(self):
        rows = L._gpu_rows(dict(Regression.AMD), 120, 1)
        for lab in ("clocks", "limits", "memory", "id"):
            self.assertFalse(_labelled(rows, lab), lab)
        self.assertNotIn("idle only", _text(rows))

    # ---- the pcie DPM table is a target, not the link
    def test_pcie_dpm_table_is_flagged_against_the_link(self):
        s = self._gpu().sample()
        rows = L._gpu_rows(s, 200)
        cells = [t for t, a in _flat(_group(rows, "clocks")) if "table x16" in t and "link x1" in t]
        self.assertTrue(cells, "no cell naming both the table's x16 and the link's x1")
        self.assertEqual(_attrs_of(_group(rows, "clocks"), "driver target"), [L.WARN])
        wide = dict(W9170_SYSFS, current_link_width="16\n")
        s2 = self._gpu("b", files=wide).sample()
        rows2 = L._gpu_rows(s2, 200)
        self.assertEqual(_attrs_of(_group(rows2, "clocks"), "driver target"), [])

    # ---- --probe prints the same data
    def test_probe_prints_the_amd_detail(self):
        s = self._gpu().sample()
        lines = self.need("_amd_probe_lines")(s)
        joined = "\n".join(lines)
        self.assertIn("vbios=113-C6812000-100", joined)
        self.assertIn("slow_age=", joined)
        self.assertIn("_amd_probe_lines(", inspect.getsource(L.probe))

# ================================================================ H. round 7: DRM clients, read-only,
# polling budget and read tiers on the legacy AMD card
#
# On 17/09 at 12:07:52 the round-6 gate opened pp_dpm_sclk while llama-bench loaded the W9170: the
# busy register read 0 in the 1-2 s gap between two repetitions and the load resumed during the slow
# read. The fdinfo texts below are verbatim from the read-only capture taken during that load
# (scratchpad drm-fdinfo-fixture, 12:13): llama-bench held two fds on renderD128 with the same
# drm-client-id, 12235116 KiB of RESIDENT VRAM and a moving drm-engine-compute counter; Xorg held
# five fds on card1 with the same client id, 0 KiB resident (10152 KiB requested, all evicted) and
# no drm-engine line at all; the fds on the NVIDIA render nodes carry no drm-* key. The kernel prints
# the engine line only for a non-zero counter (amdgpu_fdinfo.c:117-123) and drm-memory-<region> is
# the resident figure (amdgpu_fdinfo.c:100-101, drm-usage-stats.rst:182-185).
LLAMA_BENCH_FDINFO = """pos:\t0
flags:\t02100002
mnt_id:\t28
ino:\t953
drm-driver:\tamdgpu
drm-client-id:\t584
drm-pdev:\t0000:23:00.0
pasid:\t36
drm-total-cpu:\t0
drm-shared-cpu:\t0
drm-resident-cpu:\t0
drm-purgeable-cpu:\t0
drm-total-gtt:\t68416 KiB
drm-shared-gtt:\t0
drm-resident-gtt:\t68416 KiB
drm-purgeable-gtt:\t0
drm-total-vram:\t12235116 KiB
drm-shared-vram:\t0
drm-resident-vram:\t12235116 KiB
drm-purgeable-vram:\t0
drm-total-gds:\t0
drm-shared-gds:\t0
drm-resident-gds:\t0
drm-purgeable-gds:\t0
drm-total-gws:\t0
drm-shared-gws:\t0
drm-resident-gws:\t0
drm-purgeable-gws:\t0
drm-total-oa:\t0
drm-shared-oa:\t0
drm-resident-oa:\t0
drm-purgeable-oa:\t0
drm-total-doorbell:\t0
drm-shared-doorbell:\t0
drm-resident-doorbell:\t0
drm-purgeable-doorbell:\t0
drm-total-mmioremap:\t0
drm-shared-mmioremap:\t0
drm-resident-mmioremap:\t0
drm-purgeable-mmioremap:\t0
drm-memory-vram:\t12235116 KiB
drm-memory-gtt: \t68416 KiB
drm-memory-cpu: \t0 KiB
amd-evicted-vram:\t0 KiB
amd-requested-vram:\t12235116 KiB
amd-requested-gtt:\t68416 KiB
drm-engine-compute:\t23017368693 ns
"""
# the second fd of the same client, read a moment later: the counter had moved
LLAMA_BENCH_FDINFO_25 = LLAMA_BENCH_FDINFO.replace("23017368693", "23021380298")
XORG_FDINFO = """pos:\t0
flags:\t02104002
mnt_id:\t719
ino:\t954
drm-driver:\tamdgpu
drm-client-id:\t38
drm-pdev:\t0000:23:00.0
pasid:\t8
drm-total-cpu:\t0
drm-shared-cpu:\t0
drm-resident-cpu:\t0
drm-purgeable-cpu:\t0
drm-total-gtt:\t4112 KiB
drm-shared-gtt:\t0
drm-resident-gtt:\t14264 KiB
drm-purgeable-gtt:\t0
drm-total-vram:\t10152 KiB
drm-shared-vram:\t0
drm-resident-vram:\t0
drm-purgeable-vram:\t0
drm-total-gds:\t0
drm-shared-gds:\t0
drm-resident-gds:\t0
drm-purgeable-gds:\t0
drm-total-gws:\t0
drm-shared-gws:\t0
drm-resident-gws:\t0
drm-purgeable-gws:\t0
drm-total-oa:\t0
drm-shared-oa:\t0
drm-resident-oa:\t0
drm-purgeable-oa:\t0
drm-total-doorbell:\t0
drm-shared-doorbell:\t0
drm-resident-doorbell:\t0
drm-purgeable-doorbell:\t0
drm-total-mmioremap:\t0
drm-shared-mmioremap:\t0
drm-resident-mmioremap:\t0
drm-purgeable-mmioremap:\t0
drm-memory-vram:\t0 KiB
drm-memory-gtt: \t14264 KiB
drm-memory-cpu: \t0 KiB
amd-evicted-vram:\t10152 KiB
amd-requested-vram:\t10152 KiB
amd-requested-gtt:\t4112 KiB
"""
NV_FDINFO = "pos:\t0\nflags:\t02100002\nmnt_id:\t28\nino:\t972\n"
W9170_PDEV = "0000:23:00.0"
# the fds of the two captured processes, as `N -> target` gave them: only the ones that matter
# for the scanner (every /dev/dri target, one file, one device node that is not DRM)
LLAMA_BENCH_PROC = ("llama-bench", {
    0: ("/dev/null", None), 10: ("/dev/dri/renderD129", NV_FDINFO),
    14: ("/dev/dri/renderD130", NV_FDINFO), 18: ("/dev/dri/renderD131", NV_FDINFO),
    24: ("/dev/dri/renderD128", LLAMA_BENCH_FDINFO), 25: ("/dev/dri/renderD128", LLAMA_BENCH_FDINFO_25),
    26: ("/mnt/disk2tb/Modelli/model.gguf", None), 27: ("/dev/nvidia1", None)})
XORG_PROC = ("Xorg", {
    0: ("/dev/tty2", None), 14: ("/dev/dri/card1", XORG_FDINFO), 17: ("/dev/dri/card1", XORG_FDINFO),
    22: ("/dev/dri/renderD130", NV_FDINFO), 23: ("/dev/dri/card3", NV_FDINFO),
    27: ("/dev/dri/renderD129", NV_FDINFO), 30: ("/dev/dri/card1", XORG_FDINFO),
    31: ("/dev/dri/renderD131", NV_FDINFO), 32: ("/dev/dri/card1", XORG_FDINFO),
    33: ("/dev/dri/card1", XORG_FDINFO), 57: ("/dev/dri/renderD129", NV_FDINFO),
    61: ("/dev/dri/renderD130", NV_FDINFO), 65: ("/dev/dri/renderD131", NV_FDINFO)})


def _fdinfo_variant(client_id, vram_kib, engine=None):
    """The llama-bench text with another client id and resident VRAM, and the engine line
    replaced (`engine` is the line without the trailing newline) or dropped when None."""
    t = (LLAMA_BENCH_FDINFO.replace("drm-client-id:\t584", f"drm-client-id:\t{client_id}")
         .replace("drm-memory-vram:\t12235116 KiB", f"drm-memory-vram:\t{vram_kib} KiB"))
    line = "drm-engine-compute:\t23017368693 ns\n"
    return t.replace(line, f"{engine}\n" if engine else "")


def drm_proc_tree(root, procs):
    """<root>/<pid>/{comm, fd/<n> -> target, fdinfo/<n>}, the three things the scanner reads
    from /proc. `procs` maps a pid to (comm, {fd: (link target, fdinfo text or None)}); a None
    text leaves fdinfo/<n> absent, the shape of a process that closed the fd between the fd
    listing and the read. The targets are dangling symlinks on purpose: readlink resolves them
    without touching /dev."""
    for pid, (comm, fds) in procs.items():
        p = os.path.join(root, str(pid))
        os.makedirs(os.path.join(p, "fd"))
        os.makedirs(os.path.join(p, "fdinfo"))
        _w(p, "comm", comm)
        for fd, (target, text) in fds.items():
            os.symlink(target, os.path.join(p, "fd", str(fd)))
            if text is not None:
                with open(os.path.join(p, "fdinfo", str(fd)), "w") as f:
                    f.write(text)
    return root


def w9170_drm_tree(tmp, **kw):
    """w9170_tree plus the two DRM nodes the kernel lists under the PCI device (the fixture's
    drm/entries: card1, controlD65, renderD128), which is how the reader learns which
    /dev/dri names are this card's."""
    card = w9170_tree(tmp, **kw)
    dev = os.path.realpath(os.path.join(card, "device"))
    for node in ("card1", "renderD128"):
        os.makedirs(os.path.join(dev, "drm", node))
    return card


class _FakeClock:
    """time.monotonic() under the test's control. Installed as the program's `time` name (its
    readers look it up at call time), so a simulated minute costs no real second and every
    interval in the program is exercised at the real tick period."""
    def __init__(self, start=1000.0):
        self.now = start

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


_ABSENT = object()


class _AmdRound7(_Base):
    """Every open of the program under test is recorded with its full path, the clock is fake,
    and /proc is a directory of this test's own: nothing here reads the machine."""
    SMC = ("pp_dpm_sclk", "pp_dpm_mclk", "freq1_input", "freq2_input", "power1_average")

    def setUp(self):
        super().setUp()
        self._opened = []
        self._forbid = ()
        real_open = open

        def trip(path, mode="r", *a, **k):
            p = str(path)
            self._opened.append(p)
            if os.path.basename(p) in self._forbid:
                raise _Touched(os.path.basename(p))
            if any(c in mode for c in "wax+"):
                raise _Touched(f"open({p!r}, {mode!r})")
            if "/dev/dri" in p:
                raise _Touched(f"open({p!r})")
            return real_open(path, mode, *a, **k)
        L.open = trip
        self.addCleanup(lambda: delattr(L, "open") if "open" in vars(L) else None)
        self.clock = _FakeClock()
        real_time = L.time
        L.time = self.clock
        self.addCleanup(setattr, L, "time", real_time)
        # an empty /proc of this test's own; the default would be the real one
        self.proc = os.path.join(self.tmp.name, "proc")
        os.makedirs(self.proc)
        saved = getattr(L, "_PROC_ROOT", _ABSENT)

        def restore():
            if saved is _ABSENT:
                if "_PROC_ROOT" in vars(L):
                    delattr(L, "_PROC_ROOT")
            else:
                L._PROC_ROOT = saved
        L._PROC_ROOT = self.proc
        self.addCleanup(restore)

    def _gpu(self, sub="", procs=None, **kw):
        """A card under <tmp>/<sub> with its own /proc, which becomes the program's: a
        second card in the same test does not inherit the first one's processes."""
        base = os.path.join(self.tmp.name, sub) if sub else self.tmp.name
        card = w9170_drm_tree(base, **kw)
        if sub:
            self.proc = os.path.join(base, "proc")
            os.makedirs(self.proc)
            L._PROC_ROOT = self.proc
        if procs:
            drm_proc_tree(self.proc, procs)
        return L.AmdGpu(card, "FirePro W9170")

    def _sample(self, g):
        try:
            return g.sample()
        except _Touched as e:
            self.fail(f"the periodic sample opened {e}")

    def _tick(self, g, seconds=1.0):
        s = self._sample(g)
        self.clock.sleep(seconds)
        return s

    def _count(self, base):
        return sum(1 for p in self._opened if os.path.basename(p) == base)

    def _fdinfo_reads(self):
        return [p for p in self._opened if "/fdinfo/" in p]

    def _set_busy(self, g, pct):
        dev = os.path.realpath(g.dev)
        for n in ("gpu_busy_percent", "mem_busy_percent"):
            with open(os.path.join(dev, n), "w") as f:
                f.write(f"{pct}\n")

    def _amd(self, s):
        self.assertIn("amd", s, "behaviour missing: AmdGpu.sample()['amd']")
        return s["amd"]


class AmdDrmClients(_AmdRound7):
    """R1/R2/R4: the SMC gate is driven by the card's DRM clients, read-only, with a reason."""

    # ---- AC-2a positive: the 12:07:52 race. The busy register reads 0 (the gap) and a
    # llama-bench-like client holds the card: nothing may be sent to the SMC
    def test_llama_bench_client_shuts_the_gate_in_a_gap(self):
        g = self._gpu(procs={77042: LLAMA_BENCH_PROC, 2122: XORG_PROC})   # fixture busy: 0 %
        self._forbid = self.SMC
        for _ in range(60):
            s = self._tick(g)
        self.assertIsNone(s.get("power"))
        self.assertIsNone(s.get("sclk"))
        self.assertIn("llama-bench holds 11.7 GiB", s["needs"]["power"])
        self.assertIn("llama-bench holds 11.7 GiB", s["needs"]["clocks"])
        a = self._amd(s)
        self.assertEqual(a.get("clients_compute"), 1, "behaviour missing: amd['clients_compute']")

    def test_alternating_load_never_opens_an_smc_file(self):
        # llama-bench repetitions: 100 % for a second, 0 % for a second, sixty times over, with
        # the client visible (first card) and with no client visible at all (second card): the
        # window closes the register race on its own, the scan closes the client case
        for sub, procs in (("", {77042: LLAMA_BENCH_PROC}), ("b", None)):
            g = self._gpu(sub, procs=procs)
            self._forbid = self.SMC
            for i in range(60):
                self._set_busy(g, 100 if i % 2 == 0 else 0)
                s = self._tick(g)
            self.assertIsNone(s.get("power"), sub)
            self.assertIn("idle only", s["needs"]["clocks"], sub)
            self.assertEqual(self._amd(s).get("clients_scans", 0), 0, sub)

    # ---- AC-2b negative: a display server that never rendered on this card is not a client
    # that can start work, and a sustained idle opens the gate again
    def test_xorg_only_client_opens_after_sustained_idle(self):
        window = self.need("_AMD_IDLE_WINDOW")
        samples = self.need("_AMD_IDLE_SAMPLES")
        self.assertEqual((window, samples), (5.0, 3))
        g = self._gpu(procs={2122: XORG_PROC})
        s = self._tick(g)                                # no busy sample seen yet: round-6 rule
        self.assertEqual(self._count("pp_dpm_sclk"), 1)
        self.assertAlmostEqual(s["power"], 43.153, places=3)
        a = self._amd(s)
        self.assertEqual(a["clients_compute"], 0)
        self.assertEqual([c["comm"] for c in a["clients"]], ["Xorg"])
        self.assertEqual(a["clients"][0]["vram_mib"], 0)
        self.assertEqual(a["clients_scans"], 1)
        # only the five fds on card1 were read; the fds on the NVIDIA nodes were never opened
        self.assertEqual(sorted(os.path.basename(p) for p in self._fdinfo_reads()),
                         ["14", "17", "30", "32", "33"])
        # one busy tick, then idle: shut for the window even with the interval elapsed
        saved = L._AMD_SLOW_INTERVAL
        self.addCleanup(setattr, L, "_AMD_SLOW_INTERVAL", saved)
        L._AMD_SLOW_INTERVAL = 0.0
        self._set_busy(g, 100)
        s = self._tick(g)                                # busy at t0
        self.assertIn("GPU busy", s["needs"]["clocks"])
        self._set_busy(g, 0)
        self._forbid = self.SMC
        for i in range(4):                               # t0+1 .. t0+4: idle, too soon
            s = self._tick(g)
            self.assertIsNone(s.get("power"), i)
            self.assertIn("idle only", s["needs"]["clocks"], i)
        self._forbid = ()
        self._opened.clear()
        s = self._tick(g)                                # t0+5: window and samples both met
        self.assertEqual(self._count("pp_dpm_sclk"), 1)
        self.assertAlmostEqual(s["power"], 43.153, places=3)
        self.assertEqual(self._amd(s)["clients_scans"], 2)

    # ---- AC-2c: one client, whatever the number of fds
    def test_clients_are_deduplicated_by_client_id(self):
        g = self._gpu(procs={77042: LLAMA_BENCH_PROC, 2122: XORG_PROC})
        a = self._amd(self._tick(g))
        self.assertIn("clients", a, "behaviour missing: amd['clients']")
        by = {c["comm"]: c for c in a["clients"]}
        self.assertEqual(sorted(by), ["Xorg", "llama-bench"])
        self.assertEqual(by["llama-bench"]["id"], 584)
        self.assertEqual(by["llama-bench"]["pid"], 77042)
        self.assertEqual(by["llama-bench"]["vram_mib"], 12235116 // 1024)
        self.assertEqual(by["llama-bench"]["engine_ns"], 23021380298)   # the larger of the two
        self.assertTrue(by["llama-bench"]["compute"])
        self.assertEqual(by["Xorg"]["id"], 38)
        self.assertFalse(by["Xorg"]["compute"])
        self.assertEqual(a["clients_compute"], 1)
        # seven fdinfo reads for two clients: the count of fds, not of clients, is the cost
        self.assertEqual(len(self._fdinfo_reads()), 7)

    # ---- AC-2d: an fd whose fdinfo carries no drm-* key is not a client, even on this node
    def test_fds_without_drm_keys_are_ignored(self):
        parse = self.need("_drm_fdinfo_parse")
        self.assertIsNone(parse(NV_FDINFO))
        self.assertIsNone(parse(""))
        self.assertEqual(parse(XORG_FDINFO)["client_id"], 38)
        self.assertEqual(parse(LLAMA_BENCH_FDINFO)["pdev"], W9170_PDEV)
        g = self._gpu(procs={77042: LLAMA_BENCH_PROC,
                             900: ("nvidia-only", {3: ("/dev/dri/renderD128", NV_FDINFO)})})
        a = self._amd(self._tick(g))
        self.assertEqual([c["comm"] for c in a["clients"]], ["llama-bench"])
        opened = sorted(os.path.basename(p) for p in self._fdinfo_reads())
        self.assertEqual(opened, ["24", "25", "3"])          # 10, 14, 18 never opened

    # ---- the rule itself, both sides, on the two things it weighs
    def test_compute_client_rule_both_sides(self):
        rule = self.need("_amd_compute_clients")
        mib = self.need("_AMD_CLIENT_VRAM_MIB")
        self.assertEqual(mib, 64)
        client = lambda **kw: dict({"id": 1, "pid": 1, "comm": "x", "vram_mib": 0,
                                    "engine_ns": 0}, **kw)
        self.assertEqual(rule([client(engine_ns=1)]), 1)              # has executed: shut
        self.assertEqual(rule([client(vram_mib=mib)]), 1)             # holds a model: shut
        self.assertEqual(rule([client(vram_mib=mib - 1)]), 0)         # Xorg-like: open
        self.assertEqual(rule([client()]), 0)
        self.assertEqual(rule([]), 0)
        self.assertEqual(rule([client(engine_ns=1), client(id=2, vram_mib=mib)]), 2)
        # the parser: an engine line at 0 ns counts as never executed, resident is what counts
        parse = self.need("_drm_fdinfo_parse")
        p = parse(_fdinfo_variant(7, 0, "drm-engine-gfx:\t0 ns"))
        self.assertEqual((p["engine_ns"], p["vram_kib"]), (0, 0))
        p = parse(XORG_FDINFO)
        self.assertEqual((p["engine_ns"], p["vram_kib"]), (0, 0))    # not 10152: evicted

    # ---- AC-2e: processes that exit mid-scan, and entries that are not processes
    def test_vanished_pid_mid_scan(self):
        scan = self.need("_drm_clients_of")
        root = drm_proc_tree(self.proc, {
            77042: LLAMA_BENCH_PROC,
            600: ("closing", {24: ("/dev/dri/renderD128", None)}),    # fdinfo gone
            700: ("gone", {})})
        os.rmdir(os.path.join(root, "700", "fd"))                     # fd dir gone
        os.makedirs(os.path.join(root, "self"))                       # not a pid
        _w(os.path.join(root, "800"), "comm", "no-fd-dir")             # a pid without fd/
        r = scan(W9170_PDEV, {"card1", "renderD128"}, root)
        self.assertEqual([c["comm"] for c in r["clients"]], ["llama-bench"])
        self.assertEqual(r["unreadable"], 0)
        self.assertEqual(r["pids"], 4)

    @unittest.skipIf(os.geteuid() == 0, "root reads every fd directory")
    def test_unreadable_pid_is_counted_not_fatal(self):
        scan = self.need("_drm_clients_of")
        root = drm_proc_tree(self.proc, {77042: LLAMA_BENCH_PROC,
                                         2: ("kthreadd", {1: ("/dev/null", None)})})
        os.chmod(os.path.join(root, "2", "fd"), 0)
        self.addCleanup(os.chmod, os.path.join(root, "2", "fd"), 0o755)
        r = scan(W9170_PDEV, {"card1", "renderD128"}, root)
        self.assertEqual([c["comm"] for c in r["clients"]], ["llama-bench"])
        self.assertEqual(r["unreadable"], 1)

    def test_no_drm_node_means_no_walk(self):
        # the tree the older tests build has no drm/<node> directory: the scanner must not
        # walk /proc looking for nothing (the default root is the real /proc)
        card = w9170_tree(self.tmp.name)
        drm_proc_tree(self.proc, {77042: LLAMA_BENCH_PROC})
        g = L.AmdGpu(card, "FirePro W9170")
        s = self._tick(g)
        a = self._amd(s)
        self.assertEqual(a.get("clients_scans"), 0)
        self.assertEqual(self._fdinfo_reads(), [])
        self.assertIn("drm", a.get("clients_why") or "")
        self.assertAlmostEqual(s["power"], 43.153, places=3)          # the register gate alone

    # ---- AC-2f: strictly read-only, enforced
    def test_read_only_tripwire(self):
        import fcntl
        import subprocess

        def forbid(name):
            def f(*a, **k):
                raise _Touched(name)
            return f
        real = {"os.open": os.open, "fcntl.ioctl": fcntl.ioctl, "os.kill": os.kill,
                "subprocess.Popen": subprocess.Popen, "subprocess.run": subprocess.run,
                "os.readlink": os.readlink}

        def guarded_os_open(path, flags, *a, **k):
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                raise _Touched(f"os.open({path!r}, {flags:#o})")
            if "/dev/dri" in str(path):
                raise _Touched(f"os.open({path!r})")
            return real["os.open"](path, flags, *a, **k)

        def guarded_readlink(path, *a, **k):
            if str(path).startswith("/dev/dri"):
                raise _Touched(f"readlink({path!r})")
            return real["os.readlink"](path, *a, **k)
        os.open, fcntl.ioctl, os.kill = guarded_os_open, forbid("ioctl"), forbid("kill")
        subprocess.Popen, subprocess.run = forbid("Popen"), forbid("run")
        os.readlink = guarded_readlink

        def restore():
            os.open, fcntl.ioctl, os.kill = real["os.open"], real["fcntl.ioctl"], real["os.kill"]
            subprocess.Popen, subprocess.run = real["subprocess.Popen"], real["subprocess.run"]
            os.readlink = real["os.readlink"]
        self.addCleanup(restore)
        try:
            g = self._gpu(procs={77042: LLAMA_BENCH_PROC, 2122: XORG_PROC})
            for i in range(60):
                self._set_busy(g, 100 if i < 30 else 0)
                g.sample()
                self.clock.sleep(1.0)
        except _Touched as e:
            self.fail(f"not read-only: {e}")
        # and the source of the AMD path never names the things it must not do (the class is
        # read method by method: the module is loaded without a sys.modules entry, which is
        # what inspect needs for a class but not for a function)
        src = "".join(inspect.getsource(v) for v in vars(L.AmdGpu).values() if inspect.isfunction(v))
        src += "".join(inspect.getsource(self.need(n)) for n in
                       ("_drm_clients_of", "_drm_fdinfo_parse", "_drm_nodes_of"))
        # ("/dev/dri" is not in this list: the scanner compares readlink TARGETS against
        # /dev/dri/<node>, which is the brief's own definition; opening one is caught above)
        for needle in ("ioctl", "os.kill", "subprocess", "ptrace"):
            self.assertNotIn(needle, src, needle)

    # ---- AC-2h: the reason, in the panel, in the existing style
    def test_reason_and_clients_cell_in_the_panel(self):
        g = self._gpu(procs={77042: LLAMA_BENCH_PROC, 2122: XORG_PROC})
        s = self._tick(g)
        rows = L._gpu_rows(s, 120)
        text = _text(rows)
        self.assertIn("idle only (SMC): llama-bench holds 11.7 GiB", text)
        self.assertNotIn("power 43", text)
        clocks = _text(_group(rows, "clocks"))
        self.assertIn("clients", clocks)
        self.assertIn("llama-bench 11.7 GiB", clocks)
        # a client that has executed but holds no VRAM: named, without a figure
        g2 = self._gpu("b", procs={5: ("vkcube", {3: ("/dev/dri/renderD128",
                                                       _fdinfo_variant(9, 0, "drm-engine-gfx:\t512 ns"))})})
        s2 = self._tick(g2)
        self.assertIn("idle only (SMC): vkcube on the GPU", s2["needs"]["clocks"])
        # the worst case for width: a 15-character comm and three compute clients, at 80 columns
        procs = {i: (f"llamaserver{i:04d}", {4: ("/dev/dri/renderD128",
                                                  _fdinfo_variant(100 + i, 123456789))})
                 for i in (1, 2, 3)}
        g3 = self._gpu("c", procs=procs)
        s3 = self._tick(g3)
        self.assertEqual(self._amd(s3)["clients_compute"], 3)
        self.assertIn("3 compute clients", s3["needs"]["clocks"])
        rows = [r for r in L._gpu_rows(s3, 78) if "idle only" in _text([r]) or "clients" in _text([r])]
        self.assertTrue(rows)
        for r in rows:
            self.assertLessEqual(L._wwidth("".join(t for t, _ in r)), 74, _text([r]))
        # --probe prints the same state
        self.assertIn("clients_compute=1", "\n".join(L._amd_probe_lines(s)))

    # ---- AC-2i: a card with gpu_metrics is untouched by any of this
    def test_navi_like_card_is_unchanged(self):
        g = self._gpu(gpu_metrics=True, busy=100, procs={77042: LLAMA_BENCH_PROC})
        for _ in range(3):
            self._opened.clear()
            s = self._tick(g)
            for f in ("pp_dpm_sclk", "pp_dpm_mclk", "power1_average"):
                self.assertEqual(self._count(f), 1, f)
        self.assertEqual((s["sclk"], s["mclk"]), (300, 1250))
        self.assertNotIn("idle only", str(s.get("needs", {})))
        self.assertEqual(self._fdinfo_reads(), [])
        self.assertEqual(self._amd(s).get("clients_scans", 0), 0)


class AmdReadTiers(_AmdRound7):
    """R3/R5: every file the AMD reader touches is declared static (once per card lifetime), slow
    (once per interval) or per tick, and a simulated minute at the real tick period counts the
    opens against that declaration. The pcie link files come from the shared pcie_chain(), which
    this round does not change; they are counted under the slow tier it is called from."""
    PCIE = ("current_link_speed", "current_link_width", "max_link_speed", "max_link_width", "vendor")

    def test_tiers_are_declared_and_disjoint(self):
        static = set(self.need("_AMD_STATIC_FILES"))
        slow = set(self.need("_AMD_SLOW_FILES"))
        tick = set(self.need("_AMD_TICK_FILES"))
        smc = set(self.need("_AMD_SMC_MESSAGE_FILES"))
        self.assertFalse(static & slow or static & tick or slow & tick or smc & (static | slow | tick))
        for f in ("vbios_version", "vbios_build", "board_info", "pp_num_states", "mem_info_vram_total",
                  "mem_info_gtt_total", "mem_info_vis_vram_total", "power1_cap_default",
                  "power1_cap_min", "power1_cap_max", "power1_label", "fan1_min", "fan1_max",
                  "pwm1_min", "pwm1_max", "temp1_crit", "temp1_crit_hyst", "temp1_emergency",
                  "temp1_label"):
            self.assertIn(f, static, f)
        for f in ("gpu_busy_percent", "mem_busy_percent", "mem_info_vram_used", "mem_info_gtt_used",
                  "temp1_input", "in0_input", "fan1_input", "pwm1"):
            self.assertIn(f, tick, f)
        for f in ("power_dpm_state", "power_dpm_force_performance_level", "pp_cur_state",
                  "pp_force_state", "pp_power_profile_mode", "pp_sclk_od", "pp_mclk_od", "pp_dpm_pcie",
                  "thermal_throttling_logging", "mem_info_vis_vram_used", "mem_info_preempt_used",
                  "power_state", "control", "runtime_status", "power1_cap", "fan1_target",
                  "pwm1_enable", "fan1_enable"):
            self.assertIn(f, slow, f)

    def _minute(self, g, busy=None):
        """Sixty ticks of one second; returns the counts of every basename opened by sample()."""
        self._opened.clear()
        for i in range(60):
            if busy is not None:
                self._set_busy(g, busy)
            self._tick(g)
        from collections import Counter
        return Counter(os.path.basename(p) for p in self._opened)

    # ---- R5 tripwire: a static file is never opened by the periodic path
    def test_static_files_are_opened_once_at_construction_and_never_again(self):
        static = self.need("_AMD_STATIC_FILES")
        g = self._gpu()
        at_init = [os.path.basename(p) for p in self._opened]
        for f in ("vbios_version", "mem_info_vram_total", "power1_cap_max", "fan1_max", "temp1_crit"):
            self.assertEqual(at_init.count(f), 1, f)
        self._forbid = tuple(static)
        self._minute(g)                                    # _sample fails on the first static open

    # ---- AC-2g: the budget, three states of the card
    def test_polling_budget_idle_without_clients(self):
        static, slow = self.need("_AMD_STATIC_FILES"), self.need("_AMD_SLOW_FILES")
        tick, smc = self.need("_AMD_TICK_FILES"), self.need("_AMD_SMC_MESSAGE_FILES")
        g = self._gpu()
        n = self._minute(g)
        for f in static:
            self.assertEqual(n[f], 0, f)
        for f in ("gpu_busy_percent", "mem_busy_percent", "mem_info_vram_used", "mem_info_gtt_used",
                  "temp1_input", "in0_input", "fan1_input", "pwm1"):
            self.assertEqual(n[f], 60, f)
        for f in slow:
            self.assertLessEqual(n[f], 6, f)
        self.assertEqual(n["power_dpm_state"], 6)
        # every ten seconds at idle; round 7c (R1): the two hwmon clocks are never read
        for f in smc:
            self.assertEqual(n[f], 0 if f in ("freq1_input", "freq2_input") else 6, f)
        for f in self.PCIE:
            self.assertLessEqual(n[f], 6 * 4, f)           # the walk, at the slow cadence
        self.assertEqual(n["gpu_metrics"], 0)              # its absence is static
        self.assertEqual(n["pp_dpm_fclk"], 0)
        self.assertEqual(n["pp_dpm_socclk"], 0)
        # every open of the minute is accounted for by one of the declared tiers
        known = set(static) | set(slow) | set(tick) | set(smc) | set(self.PCIE)
        self.assertEqual(sorted(set(n) - known), [])
        a = self._amd(self._tick(g))
        self.assertLessEqual(a["clients_scans"], 7)
        self.assertEqual(self._fdinfo_reads(), [])
        self.assertLess(sum(n.values()), 900, "opens per idle minute; round 6 measured 3480")

    def test_polling_budget_idle_with_a_display_server(self):
        g = self._gpu(procs={2122: XORG_PROC})
        n = self._minute(g)
        self.assertEqual(n["pp_dpm_sclk"], 6)
        self.assertEqual(len(self._fdinfo_reads()), 6 * 5)   # five fds on card1, once per scan
        self.assertEqual(self._amd(self._tick(g))["clients_scans"], 7)

    def test_polling_budget_under_load(self):
        g = self._gpu(procs={77042: LLAMA_BENCH_PROC, 2122: XORG_PROC})
        n = self._minute(g, busy=100)
        for f in self.SMC:
            self.assertEqual(n[f], 0, f)
        self.assertEqual(self._fdinfo_reads(), [])
        a = self._amd(self._tick(g))
        self.assertEqual(a.get("clients_scans"), 0, "behaviour missing: amd['clients_scans']")
        self.assertEqual(n["gpu_busy_percent"], 60)
        self.assertEqual(n["power_dpm_state"], 6)          # the ungated slow tier still runs

    def test_polling_budget_idle_with_a_model_holder(self):
        # an idle llama-server: the card reads 0 %, the scan finds the client every ten seconds
        # and nothing is sent to the SMC
        g = self._gpu(procs={77042: LLAMA_BENCH_PROC})
        n = self._minute(g)
        for f in self.SMC:
            self.assertEqual(n[f], 0, f)
        self.assertEqual(len(self._fdinfo_reads()), 6 * 2)
        self.assertEqual(self._amd(self._tick(g))["clients_scans"], 7)

    def test_slow_values_and_static_values_survive_between_reads(self):
        g = self._gpu()
        first = self._amd(self._tick(g))
        for _ in range(5):
            later = self._amd(self._tick(g))
        for k in ("vbios", "board", "fw", "states", "power_cap_default", "fan_max", "temp_crit",
                  "perf_level", "dpm_state", "cur_state", "pcie_levels", "vis_vram_used"):
            self.assertEqual(later.get(k), first.get(k), k)
        self.assertEqual(later["missing"], [])


# ------------------------------------------------- round 7b: the VulkanN <-> PCI map from files
# The four cards of the capture machine as sysfs shows them (fixture sysfs-cards.txt, 17/09):
# (card, pci, vendor, device, class, boot_vga or None when the file is absent, driver, nodes)
VK_CARDS = (
    ("card1", "0000:23:00.0", "0x1002", "0x67a0", "0x038000", None, "amdgpu", ("card1", "renderD128")),
    ("card2", "0000:12:00.0", "0x10de", "0x15f8", "0x030200", None, "nvidia", ("card2", "renderD129")),
    ("card3", "0000:14:00.0", "0x10de", "0x1f08", "0x030000", "1", "nvidia", ("card3", "renderD130")),
    ("card4", "0000:1b:00.0", "0x10de", "0x15f8", "0x030200", None, "nvidia", ("card4", "renderD131")),
)
# ground truth measured on that machine: vulkaninfo, MESA_VK_DEVICE_SELECT=list and
# llama-bench --list-devices agree (fixture vulkan-order-fixture/, 17/09 13:16)
VK_GROUND = {"0000:14:00.0": 0, "0000:12:00.0": 1, "0000:1b:00.0": 2, "0000:23:00.0": 3}
VK_LOADER_ORDER = {"0000:12:00.0": 0, "0000:14:00.0": 1, "0000:1b:00.0": 2, "0000:23:00.0": 3}
# the nine ICD manifests installed there (fixture icd.d.txt), file stem -> library_path
VK_ICDS = {"asahi_icd": "libvulkan_asahi.so", "gfxstream_vk_icd": "libvulkan_gfxstream.so",
           "intel_hasvk_icd": "libvulkan_intel_hasvk.so", "intel_icd": "libvulkan_intel.so",
           "lvp_icd": "libvulkan_lvp.so", "nouveau_icd": "libvulkan_nouveau.so",
           "nvidia_icd": "libGLX_nvidia.so.0", "radeon_icd": "libvulkan_radeon.so",
           "virtio_icd": "libvulkan_virtio.so"}
# the two implicit-layer manifests, verbatim (fixture VkLayer_MESA_device_select.json and
# nvidia_layers.json): the loader's single-layer and multi-layer forms
VK_LAYER_DEVSEL = """{
  "file_format_version" : "1.0.0",
  "layer" : {
    "name": "VK_LAYER_MESA_device_select",
    "type": "GLOBAL",
    "library_path": "libVkLayer_MESA_device_select.so",
    "api_version": "1.4.303",
    "implementation_version": "1",
    "description": "Linux device selection layer",
    "functions": {
      "vkNegotiateLoaderLayerInterfaceVersion": "vkNegotiateLoaderLayerInterfaceVersion"
    },
    "disable_environment": {
      "NODEVICE_SELECT": "1"
    }
  }
}
"""
VK_LAYER_NV = """{
    "file_format_version" : "1.0.1",
    "layers": [{
        "name": "VK_LAYER_NV_optimus",
        "type": "INSTANCE",
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.4.312",
        "implementation_version" : "1",
        "description" : "NVIDIA Optimus layer",
        "functions": {
            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",
            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"
        },
        "enable_environment": {
            "__NV_PRIME_RENDER_OFFLOAD": "1"
        },
        "disable_environment": {
            "DISABLE_LAYER_NV_OPTIMUS_1": ""
        }
    },{
        "name": "VK_LAYER_NV_present",
        "type": "INSTANCE",
        "library_path": "libnvidia-present.so.580.173.02",
        "api_version" : "1.4.312",
        "implementation_version" : "1",
        "description" : "NVIDIA GR2608 layer",
        "functions": {
            "vkGetInstanceProcAddr": "vk_nvpGetInstanceProcAddr",
            "vkGetDeviceProcAddr": "vk_nvpGetDeviceProcAddr"
        },
        "enable_environment": {
            "NVPRESENT_ENABLE_SMOOTH_MOTION": "1"
        },
        "disable_environment": {
            "DISABLE_LAYER_NV_GR2608_1": ""
        }
    }]
}
"""
VK_LAYERS = {"VkLayer_MESA_device_select": VK_LAYER_DEVSEL, "nvidia_layers": VK_LAYER_NV}
# the session environment of the capture (fixture session-env.txt); HOME points nowhere on
# purpose, so the loader's ~/.config and ~/.local/share directories are searched and empty
VK_ENV = {"XDG_SESSION_TYPE": "x11", "DISPLAY": ":0", "HOME": "/home/nobody"}


def vk_cards_tree(drm, pci_root, cards=VK_CARDS, w9170=None):
    """<drm>/<card>/device -> <pci_root>/<pci> holding vendor, device, class, boot_vga (only
    when given: the kernel creates the file on VGA-class devices alone), a driver link to a
    directory named after the driver, and drm/<node> entries. `w9170` names an existing PCI
    device directory (w9170_tree's) to use for card1 instead of a bare one, so discover_gpus
    can build a real AmdGpu on it."""
    for card, pci, vendor, device, cls, boot_vga, driver, nodes in cards:
        dev = w9170 if (w9170 and card == "card1") else os.path.join(pci_root, pci)
        os.makedirs(dev, exist_ok=True)
        for name, content in (("vendor", vendor), ("device", device), ("class", cls)):
            _w(dev, name, content)
        if boot_vga is not None:
            _w(dev, "boot_vga", boot_vga)
        drv = os.path.join(pci_root, "drivers", driver)
        os.makedirs(drv, exist_ok=True)
        os.symlink(drv, os.path.join(dev, "driver"))
        for n in nodes:
            os.makedirs(os.path.join(dev, "drm", n), exist_ok=True)
        os.makedirs(os.path.join(drm, card), exist_ok=True)
        os.symlink(dev, os.path.join(drm, card, "device"))
    return drm


def vk_manifest_tree(root, icds=VK_ICDS, layers=VK_LAYERS, where="usr/share"):
    """<root>/<where>/vulkan/icd.d/<stem>.json and implicit_layer.d/<stem>.json, the loader's
    search path with the fallback data dir /usr/share prefixed by the test's root."""
    icd_d = os.path.join(root, where, "vulkan", "icd.d")
    lay_d = os.path.join(root, where, "vulkan", "implicit_layer.d")
    os.makedirs(icd_d, exist_ok=True)
    os.makedirs(lay_d, exist_ok=True)
    for stem, lib in icds.items():
        with open(os.path.join(icd_d, f"{stem}.json"), "w") as f:
            f.write('{"ICD": {"api_version": "1.4.318", "library_path": "%s"}, '
                    '"file_format_version": "1.0.1"}\n' % lib)
    for stem, text in layers.items():
        with open(os.path.join(lay_d, f"{stem}.json"), "w") as f:
            f.write(text)
    return root


def environ_file(proc, pid, env):
    """<proc>/<pid>/environ as the kernel serves it: KEY=VALUE records ended by NUL."""
    p = os.path.join(proc, str(pid))
    os.makedirs(p, exist_ok=True)
    with open(os.path.join(p, "environ"), "wb") as f:
        f.write(b"".join(f"{k}={v}".encode() + b"\0" for k, v in env.items()))
    return p


class VulkanOrder(_Base):
    """Round 7b: the VulkanN <-> PCI map is computed from sysfs identity, the loader's manifest
    files and a process's environ, reproducing loader -> device-select -> ggml, and never runs
    a program or opens a device node. Every open of the program under test is recorded, /sys
    and /proc are trees of this test's own, and the loader's search path is rooted here."""

    def setUp(self):
        super().setUp()
        self._opened = []
        real_open = open

        def trip(path, mode="r", *a, **k):
            p = str(path)
            self._opened.append(p)
            if any(c in mode for c in "wax+"):
                raise _Touched(f"open({p!r}, {mode!r})")
            if p.startswith("/dev/dri") or p.startswith("/dev/nvidia"):
                raise _Touched(f"open({p!r})")
            return real_open(path, mode, *a, **k)
        L.open = trip
        self.addCleanup(lambda: delattr(L, "open") if "open" in vars(L) else None)
        self.clock = _FakeClock()
        real_time = L.time
        L.time = self.clock
        self.addCleanup(setattr, L, "time", real_time)
        self.drm = os.path.join(self.tmp.name, "drm")
        self.pci = os.path.join(self.tmp.name, "pci")
        self.vk = os.path.join(self.tmp.name, "vkroot")
        self.proc = os.path.join(self.tmp.name, "proc")
        for d in (self.drm, self.pci, self.vk, self.proc):
            os.makedirs(d)
        for name, val in (("_DRM_ROOT", self.drm), ("_VK_ROOT", self.vk), ("_PROC_ROOT", self.proc)):
            saved = getattr(L, name, _ABSENT)
            setattr(L, name, val)
            self.addCleanup(self._restore, name, saved)
        # the two caches are module state; every test starts with them empty
        for name, empty in (("_vulkan_map", None), ("_vulkan_pid_maps", {})):
            saved = getattr(L, name, _ABSENT)
            setattr(L, name, empty)
            self.addCleanup(self._restore, name, saved)

    @staticmethod
    def _restore(name, saved):
        if saved is _ABSENT:
            if name in vars(L):
                delattr(L, name)
        else:
            setattr(L, name, saved)

    def _resolve(self, env=None, cards=VK_CARDS, icds=VK_ICDS, layers=VK_LAYERS, sub=""):
        """The map for `env` on a machine holding `cards` and these manifests: fail (not error)
        with a readable message while the resolver is missing."""
        base = os.path.join(self.tmp.name, sub) if sub else self.tmp.name
        drm, pci, vk = (os.path.join(base, n) for n in ("drm", "pci", "vkroot"))
        if sub:
            for d in (drm, pci, vk):
                os.makedirs(d)
            L._DRM_ROOT, L._VK_ROOT = drm, vk
        vk_cards_tree(drm, pci, cards)
        vk_manifest_tree(vk, icds, layers)
        return self.need("_vulkan_resolve")(dict(VK_ENV, **(env or {})))

    def _map_fn(self):
        """vulkan_device_map taking the environment: fail (not error) while it does not."""
        fn = self.need("vulkan_device_map")
        self.assertIn("env", inspect.signature(fn).parameters, "behaviour missing: vulkan_device_map(env)")
        return fn

    def _map(self, **kw):
        r = self._resolve(**kw)
        self.assertIsNone(r.get("why"), r)
        return r["map"]

    def _why(self, **kw):
        r = self._resolve(**kw)
        self.assertIsNone(r["map"], r)
        self.assertTrue(r.get("why"), r)
        return r["why"]

    # ---- AC-2a: the fixture gives the ground truth
    def test_fixture_gives_the_ground_truth_map(self):
        r = self._resolve()
        self.assertEqual(r["map"], VK_GROUND)
        self.assertIsNone(r["why"])
        # what the map rests on is declared beside it, not hidden
        self.assertTrue(any("ggml_vk_device_is_supported" in n for n in r["notes"]), r["notes"])

    # ---- AC-2c: GGML_VK_VISIBLE_DEVICES, indices into the device-select order, in that order
    def test_ggml_visible_devices_is_honoured(self):
        self.assertEqual(self._map(env={"GGML_VK_VISIBLE_DEVICES": "3,1"}),
                         {"0000:23:00.0": 0, "0000:12:00.0": 1})
        self.assertEqual(self._map(env={"GGML_VK_VISIBLE_DEVICES": "1 3"}, sub="b"),
                         {"0000:12:00.0": 0, "0000:23:00.0": 1})
        # index 4 is llvmpipe: a valid selection that names no card
        self.assertEqual(self._map(env={"GGML_VK_VISIBLE_DEVICES": "4"}, sub="c"), {})
        # out of range: ggml throws and the process never starts, so there is no index
        self.assertIn("GGML_VK_VISIBLE_DEVICES", self._why(env={"GGML_VK_VISIBLE_DEVICES": "9"}, sub="d"))

    # ---- AC-2d: MESA_VK_DEVICE_SELECT and DRI_PRIME, reproduced from the layer's source
    def test_mesa_device_select_is_reproduced(self):
        # vid:did, the LAST match (device_select_layer.c:282-287): the W9170 goes first
        self.assertEqual(self._map(env={"MESA_VK_DEVICE_SELECT": "1002:67a0"}),
                         {"0000:23:00.0": 0, "0000:12:00.0": 1, "0000:14:00.0": 2, "0000:1b:00.0": 3})
        # two P100 share 10de:15f8: the last one in loader order, 1b, wins
        self.assertEqual(self._map(env={"MESA_VK_DEVICE_SELECT": "10de:15f8"}, sub="b"),
                         {"0000:1b:00.0": 0, "0000:12:00.0": 1, "0000:14:00.0": 2, "0000:23:00.0": 3})
        # a trailing ! exposes the default alone
        self.assertEqual(self._map(env={"MESA_VK_DEVICE_SELECT": "1002:67a0!"}, sub="c"), {"0000:23:00.0": 0})
        self.assertEqual(self._map(env={"MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE": "1"}, sub="d"),
                         {"0000:14:00.0": 0})
        # no card matches: the selection is ignored and the boot_vga rule decides
        self.assertEqual(self._map(env={"MESA_VK_DEVICE_SELECT": "8086:0001"}, sub="e"), VK_GROUND)

    def test_dri_prime_is_reproduced(self):
        # DRI_PRIME=1: the default (the 2060, loader index 1) is skipped for the first other
        # non-CPU device, the P100 at 12 (device_select_layer.c:535-547, 415-432)
        self.assertEqual(self._map(env={"DRI_PRIME": "1"}),
                         {"0000:12:00.0": 0, "0000:14:00.0": 1, "0000:1b:00.0": 2, "0000:23:00.0": 3})
        self.assertEqual(self._map(env={"DRI_PRIME": "2"}, sub="b"),
                         {"0000:1b:00.0": 0, "0000:12:00.0": 1, "0000:14:00.0": 2, "0000:23:00.0": 3})
        # the pci tag, with and without the ! suffix
        self.assertEqual(self._map(env={"DRI_PRIME": "pci-0000_23_00_0"}, sub="c"),
                         {"0000:23:00.0": 0, "0000:12:00.0": 1, "0000:14:00.0": 2, "0000:1b:00.0": 3})
        self.assertEqual(self._map(env={"DRI_PRIME": "pci-0000_23_00_0!"}, sub="d"), {"0000:23:00.0": 0})
        # vid:did through DRI_PRIME, last match as above
        self.assertEqual(self._map(env={"DRI_PRIME": "1002:67a0"}, sub="e"),
                         {"0000:23:00.0": 0, "0000:12:00.0": 1, "0000:14:00.0": 2, "0000:1b:00.0": 3})
        # DRI_PRIME=0 is the default itself
        self.assertEqual(self._map(env={"DRI_PRIME": "0"}, sub="f"), VK_GROUND)

    def test_layers_and_loader_variables_that_cannot_be_reproduced_give_a_reason(self):
        # the Optimus layer is enabled by the exact value 1 (loader.c:952-961): unmappable
        self.assertIn("VK_LAYER_NV_optimus", self._why(env={"__NV_PRIME_RENDER_OFFLOAD": "1"}))
        # and NOT by any other value: the negative control
        self.assertEqual(self._map(env={"__NV_PRIME_RENDER_OFFLOAD": "0"}, sub="b"), VK_GROUND)
        self.assertIn("VK_LAYER_NV_present", self._why(env={"NVPRESENT_ENABLE_SMOOTH_MOTION": "1"}, sub="c"))
        # the loader's own sort switched off: the order is the drivers' own, not modelled
        self.assertIn("VK_LOADER_DISABLE_SELECT", self._why(env={"VK_LOADER_DISABLE_SELECT": "1"}, sub="d"))
        self.assertEqual(self._map(env={"VK_LOADER_DISABLE_SELECT": "0"}, sub="e"), VK_GROUND)
        # an explicit layer forced on could reorder anything
        self.assertIn("VK_INSTANCE_LAYERS", self._why(env={"VK_INSTANCE_LAYERS": "VK_LAYER_KHRONOS_validation"}, sub="f"))
        # the loader's default device: one match is reproducible (it sorts first, then
        # device-select still moves the boot_vga card ahead of it)
        self.assertEqual(self._map(env={"VK_LOADER_DEVICE_SELECT": "1002:67a0"}, sub="g"),
                         {"0000:14:00.0": 0, "0000:23:00.0": 1, "0000:12:00.0": 2, "0000:1b:00.0": 3})
        # two matches: the loader takes the FIRST in ICD order (loader_linux.c:219-226), unknown
        self.assertIn("VK_LOADER_DEVICE_SELECT", self._why(env={"VK_LOADER_DEVICE_SELECT": "10de:15f8"}, sub="h"))

    def test_device_select_layer_disabled_leaves_the_loader_order(self):
        # its own disable_environment (the manifest's), then the loader's filter
        self.assertEqual(self._map(env={"NODEVICE_SELECT": "1"}), VK_LOADER_ORDER)
        self.assertEqual(self._map(env={"VK_LOADER_LAYERS_DISABLE": "~implicit~"}, sub="b"), VK_LOADER_ORDER)
        self.assertEqual(self._map(env={"VK_LOADER_LAYERS_DISABLE": "*device_select*"}, sub="c"), VK_LOADER_ORDER)
        # a filter that names another layer leaves device-select on
        self.assertEqual(self._map(env={"VK_LOADER_LAYERS_DISABLE": "VK_LAYER_NV_optimus"}, sub="d"), VK_GROUND)
        # and with no layer manifest at all the loader order is what ggml sees
        self.assertEqual(self._map(layers={}, sub="e"), VK_LOADER_ORDER)

    # ---- AC-2e: a card whose driver has no installed ICD is not indexed
    def test_card_without_an_icd_is_not_indexed(self):
        radeon = tuple(("card1", p, v, d, c, b, "radeon", n) if card == "card1" else (card, p, v, d, c, b, drv, n)
                       for card, p, v, d, c, b, drv, n in VK_CARDS)
        self.assertEqual(self._map(cards=radeon), {"0000:14:00.0": 0, "0000:12:00.0": 1, "0000:1b:00.0": 2})
        # amdgpu without its ICD manifest: the same
        icds = {k: v for k, v in VK_ICDS.items() if k != "radeon_icd"}
        self.assertEqual(self._map(icds=icds, sub="b"), {"0000:14:00.0": 0, "0000:12:00.0": 1, "0000:1b:00.0": 2})
        # amdgpu with two ICDs claiming it (RADV and AMDVLK): ggml de-duplicates by UUID and
        # may re-append the survivor, not reproducible from files
        both = dict(VK_ICDS, amd_icd64="amdvlk64.so")
        self.assertIn("amdvlk64.so", self._why(icds=both, sub="c"))
        # the driver filter removes one of them: reproducible again
        self.assertEqual(self._map(icds=both, env={"VK_LOADER_DRIVERS_DISABLE": "amd_icd64*"}, sub="d"), VK_GROUND)
        self.assertEqual(self._map(icds=both, env={"VK_LOADER_DRIVERS_SELECT": "radeon*,nvidia*"}, sub="e"), VK_GROUND)
        # VK_DRIVER_FILES replaces the search altogether
        only = os.path.join(self.tmp.name, "f", "vkroot", "usr/share/vulkan/icd.d/nvidia_icd.json")
        self.assertEqual(self._map(env={"VK_DRIVER_FILES": only}, sub="f"),
                         {"0000:14:00.0": 0, "0000:12:00.0": 1, "0000:1b:00.0": 2})

    # ---- R2 step 1, the loader's type order: an integrated GPU sorts after every discrete one
    # whatever its bus (loader_linux.c:38-56, 84-90). Sysfs has no type flag: an i915/xe card on
    # bus 0 is taken as integrated (where Intel puts the iGPU), any other card as discrete.
    def test_integrated_intel_gpu_sorts_after_discrete_cards(self):
        igpu = ("card0", "0000:00:02.0", "0x8086", "0x9a49", "0x030000", None, "i915", ("card0", "renderD132"))
        cards = (igpu,) + tuple((card, p, v, d, c, None, drv, n) for card, p, v, d, c, b, drv, n in VK_CARDS)
        # no boot_vga anywhere, so the default is the first device in loader order and
        # DRI_PRIME=1 skips it: with the iGPU last, index 0 is the P100 at 12 and the skip
        # lands on the 2060; with the iGPU first (by bus alone) the skip would land on 12
        self.assertEqual(self._map(cards=cards, env={"DRI_PRIME": "1"}),
                         {"0000:14:00.0": 0, "0000:12:00.0": 1, "0000:1b:00.0": 2, "0000:23:00.0": 3,
                          "0000:00:02.0": 4})
        # the same Intel part on a non-zero bus is a discrete card and sorts by address
        arc = ("card0", "0000:03:00.0", "0x8086", "0x56a0", "0x030000", None, "i915", ("card0", "renderD132"))
        cards = (arc,) + cards[1:]
        self.assertEqual(self._map(cards=cards, env={"DRI_PRIME": "1"}, sub="b"),
                         {"0000:12:00.0": 0, "0000:03:00.0": 1, "0000:14:00.0": 2, "0000:1b:00.0": 3,
                          "0000:23:00.0": 4})

    # ---- AC-2f: the CPU device is never indexed
    def test_cpu_device_is_not_indexed(self):
        self.assertEqual(self._map(), VK_GROUND)                       # lvp installed
        icds = {k: v for k, v in VK_ICDS.items() if k != "lvp_icd"}
        self.assertEqual(self._map(icds=icds, sub="b"), VK_GROUND)      # and not
        # llvmpipe alone: ggml finds no GPU and starts with none (ggml-vulkan.cpp:7930-7942)
        self.assertEqual(self._map(cards=(), sub="c"), {})

    # ---- AC-2g: no boot_vga card and DISPLAY set: reproduced, because ggml enables no surface
    # extension and the layer consults the X server only for one that does
    def test_no_boot_vga_card_with_a_display_is_reproduced_not_guessed(self):
        headless = tuple((card, p, v, d, "0x030200", None, drv, n) for card, p, v, d, c, b, drv, n in VK_CARDS)
        r = self._resolve(cards=headless, env={"DISPLAY": ":0", "WAYLAND_DISPLAY": "wayland-0"})
        self.assertEqual(r["map"], VK_LOADER_ORDER)
        self.assertIsNone(r["why"])
        self.assertTrue(any("surface" in n for n in r["notes"]), r["notes"])
        # positive control of the rule itself: boot_vga on another card moves that card first
        vga_1b = tuple((card, p, v, d, c, ("1" if p == "0000:1b:00.0" else None), drv, n)
                       for card, p, v, d, c, b, drv, n in headless)
        self.assertEqual(self._map(cards=vga_1b, sub="b"),
                         {"0000:1b:00.0": 0, "0000:12:00.0": 1, "0000:14:00.0": 2, "0000:23:00.0": 3})
        # boot_vga=0 on a VGA-class card is not boot_vga
        vga_0 = tuple((card, p, v, d, c, ("0" if p == "0000:14:00.0" else None), drv, n)
                      for card, p, v, d, c, b, drv, n in headless)
        self.assertEqual(self._map(cards=vga_0, sub="c"), VK_LOADER_ORDER)

    # ---- AC-2h: the map reads at discovery, or once per pid, and never again
    def test_map_reads_only_at_discovery_or_once_per_pid(self):
        vk_cards_tree(self.drm, self.pci)
        vk_manifest_tree(self.vk)
        environ_file(self.proc, 77042, {"GGML_VK_VISIBLE_DEVICES": "3"})
        first = self._map_fn()(dict(VK_ENV))
        self.assertEqual(first, VK_GROUND)
        self.assertTrue(self._opened, "the first call read nothing")
        self._opened.clear()
        for _ in range(60):
            self.assertEqual(L.vulkan_device_map(dict(VK_ENV)), VK_GROUND)
        self.assertEqual(self._opened, [], "the cached map re-read a file")
        r = self.need("vulkan_map_for_pid")("77042")
        self.assertEqual(r["map"], {"0000:23:00.0": 0})
        env_reads = [p for p in self._opened if p.endswith("/environ")]
        self.assertEqual(len(env_reads), 1)
        for _ in range(59):
            self.assertEqual(L.vulkan_map_for_pid("77042")["map"], {"0000:23:00.0": 0})
        self.assertEqual(len([p for p in self._opened if p.endswith("/environ")]), 1)
        # an exited pid is forgotten, a returning one is read again
        self.need("_vulkan_forget_pids")(set())
        self.assertEqual(L._vulkan_pid_maps, {})
        L.vulkan_map_for_pid("77042")
        self.assertEqual(len([p for p in self._opened if p.endswith("/environ")]), 2)
        L._vulkan_forget_pids({"77042"})
        self.assertIn("77042", L._vulkan_pid_maps)

    def test_unreadable_environ_is_unmappable_with_a_reason(self):
        vk_cards_tree(self.drm, self.pci)
        vk_manifest_tree(self.vk)
        # no such process (ENOENT / ESRCH)
        r = self.need("vulkan_map_for_pid")("4242")
        self.assertIsNone(r["map"])
        self.assertIn("environ", r["why"])
        # a focused port with no local process carries "?" as its pid (sample_llama_fleet)
        r = L.vulkan_map_for_pid("?")
        self.assertIsNone(r["map"])
        self.assertTrue(r["why"])
        if os.geteuid() != 0:
            p = environ_file(self.proc, 4343, {})
            os.chmod(os.path.join(p, "environ"), 0)
            self.addCleanup(os.chmod, os.path.join(p, "environ"), 0o600)
            r = L.vulkan_map_for_pid("4343")
            self.assertIsNone(r["map"])
            self.assertIn("denied", r["why"].lower())

    # ---- AC-2i: the consumers, on this machine's tree
    def _machine(self):
        """The full tree discover_gpus walks: the W9170 fixture as card1 (a real AmdGpu is built
        on it), bare NVIDIA cards, manifests, and every way to start a program forbidden for the
        rest of the test (round 7c, R3): discovery must execute none, whatever PATH holds. Until
        round 7c this hid lspci from PATH instead, which only proved the program was not found."""
        w = w9170_drm_tree(os.path.join(self.tmp.name, "w"))
        dev = os.path.realpath(os.path.join(w, "device"))
        vk_cards_tree(self.drm, self.pci, w9170=dev)
        vk_manifest_tree(self.vk)
        # no pci.ids either: the names are not asserted here, and no host file is read
        saved_paths, cache = L._PCI_IDS_PATHS, self.need("_pci_ids_cache")
        saved_cache = dict(cache)
        L._PCI_IDS_PATHS = ()
        cache.clear()
        self.addCleanup(lambda: (setattr(L, "_PCI_IDS_PATHS", saved_paths), cache.clear(), cache.update(saved_cache)))
        forbid_programs(self)
        try:
            return self.need("discover_gpus")()
        except _Touched as e:
            self.fail(f"discovery executed a program: {e}")

    def _collect_power(self, llama_env, devs):
        """_collect's power line of one server, pid 77042 with `llama_env`, `-dev` = devs; the
        readers _collect touches beyond the GPUs are stubbed so nothing of the machine is read."""
        from types import SimpleNamespace
        gpus, _feeds = self._machine()
        environ_file(self.proc, 77042, llama_env)
        stubs = {"cpu_sample": lambda: {"util": 0.0}, "mem_sample": lambda: {"free": 1, "total": 2},
                 "discover_hwmon_extra": lambda exclude=(): [], "llama_processes": lambda: [],
                 "llama_devices_from_cmdline": lambda port=None: list(devs),
                 "llama_settings_from_cmdline": lambda port=None: None}
        for name, fn in stubs.items():
            real = getattr(L, name)
            setattr(L, name, fn)
            self.addCleanup(setattr, L, name, real)
        # every sample carries a power figure so the attribution is arithmetic, not luck
        for g in gpus:
            real_sample = g.sample
            g.sample = (lambda rs=real_sample, pci=g.pci_addr: dict(rs(), power=float(int(pci[5:7], 16))))
        feed = SimpleNamespace(data=[{"pid": "77042", "port": "8080", "alive": True, "model": "m", "phase": "idle"}])
        out = L._collect(gpus, None, feed)
        return gpus, out[3][0]["power_w"]

    def test_consumers_labels_and_power_attribution(self):
        gpus, _ = self._machine()
        self.assertEqual([g.pci_addr for g in gpus], ["0000:23:00.0", "0000:12:00.0", "0000:14:00.0", "0000:1b:00.0"])
        self.assertEqual([g.backend_dev for g in gpus], ["Vulkan3", "Vulkan1", "Vulkan0", "Vulkan2"])
        # -dev Vulkan3 in the fixture environment is the W9170 alone: bus 0x23 = 35 W
        self.tmp.cleanup(); self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.drm, self.pci, self.vk, self.proc = (os.path.join(self.tmp.name, n) for n in ("drm", "pci", "vkroot", "proc"))
        for d in (self.drm, self.pci, self.vk, self.proc):
            os.makedirs(d)
        L._DRM_ROOT, L._VK_ROOT, L._PROC_ROOT, L._vulkan_map, L._vulkan_pid_maps = self.drm, self.vk, self.proc, None, {}
        _, w = self._collect_power(VK_ENV, ["Vulkan3"])
        self.assertEqual(w, 35.0)

    def test_attribution_follows_the_process_environment_not_the_panel(self):
        # the server was started with GGML_VK_VISIBLE_DEVICES=3: ITS Vulkan0 is the W9170,
        # while the panel label, from llamagputop's own environment, still says Vulkan3
        gpus, w = self._collect_power(dict(VK_ENV, GGML_VK_VISIBLE_DEVICES="3"), ["Vulkan0"])
        self.assertEqual(w, 35.0)
        self.assertEqual(gpus[0].backend_dev, "Vulkan3")

    # ---- AC-2j: negative control, a machine with no Vulkan ICD at all
    def test_no_icd_gives_an_empty_map_and_the_all_cards_fallback(self):
        r = self._resolve(icds={}, layers={})
        self.assertEqual(r["map"], {})
        self.assertIn("ICD", r["why"] or "")
        self.tmp.cleanup(); self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.drm, self.pci, self.vk, self.proc = (os.path.join(self.tmp.name, n) for n in ("drm", "pci", "vkroot", "proc"))
        for d in (self.drm, self.pci, self.vk, self.proc):
            os.makedirs(d)
        L._DRM_ROOT, L._VK_ROOT, L._PROC_ROOT, L._vulkan_map, L._vulkan_pid_maps = self.drm, self.vk, self.proc, None, {}
        real_tree = vk_manifest_tree

        def no_manifests(root, icds=None, layers=None, where="usr/share"):
            return real_tree(root, {}, {}, where)
        globals()["vk_manifest_tree"] = no_manifests
        self.addCleanup(globals().__setitem__, "vk_manifest_tree", real_tree)
        gpus, w = self._collect_power(VK_ENV, ["Vulkan3"])
        self.assertEqual([g.backend_dev for g in gpus], [None] * 4)
        self.assertEqual(w, 35.0 + 18.0 + 20.0 + 27.0)                  # -dev set but unmappable: all cards

    # ---- the reason reaches --probe
    def test_probe_prints_the_map_and_its_reason(self):
        import contextlib
        import io
        gpus, _ = self._machine()
        environ_file(self.proc, 77042, VK_ENV)
        stubs = {"discover_gpus": lambda: (gpus, {}),
                 "cpu_sample": lambda: {"name": "cpu", "util": 0.0, "temp": None, "freq": None, "power": None,
                                        "rapl_present": False, "loadavg": "0", "temps": {}, "temp_limits": {},
                                        "temp_relative": set()},
                 "mem_sample": lambda: {"free": 1, "total": 2, "swap_used": 0, "swap_total": 0},
                 "discover_hwmon_extra": lambda exclude=(): [],
                 "sample_llama_fleet": lambda port=None: [
                     {"pid": "77042", "port": "8080", "alive": True, "model": "m", "phase": "idle", "spec": None,
                      "spec_head": None, "spec_type": None, "ctx": None, "ctx_total": None, "slots": None,
                      "metrics_off": False}],
                 "llama_devices_from_cmdline": lambda port=None: ["Vulkan3"]}
        for name, fn in stubs.items():
            real = getattr(L, name)
            setattr(L, name, fn)
            self.addCleanup(setattr, L, name, real)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            L.probe()
        text = buf.getvalue()
        self.assertIn("vulkan: Vulkan3", text)
        self.assertIn("vulkan: Vulkan0", text)
        self.assertIn("dev=Vulkan3", text)
        self.assertIn("0000:23:00.0", text)
        # and an unmappable process says why, in the same line
        L._vulkan_pid_maps.clear()
        environ_file(self.proc, 77042, dict(VK_ENV, __NV_PRIME_RENDER_OFFLOAD="1"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            L.probe()
        self.assertIn("VK_LAYER_NV_optimus", buf.getvalue())

    # ---- AC-2b: no program, no device node, no Vulkan/GL library, ever
    def test_read_only_tripwire_no_program_no_device_node(self):
        import ast
        import ctypes
        import subprocess

        def forbid(name):
            def f(*a, **k):
                raise _Touched(name)
            return f
        names = ("execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe",
                 "posix_spawn", "posix_spawnp", "system", "popen", "fork")
        real = {n: getattr(os, n) for n in names if hasattr(os, n)}
        real_os_open, real_popen, real_run, real_cdll = os.open, subprocess.Popen, subprocess.run, ctypes.CDLL

        def guarded_os_open(path, flags, *a, **k):
            p = str(path)
            if p.startswith("/dev/dri") or p.startswith("/dev/nvidia"):
                raise _Touched(f"os.open({p!r})")
            return real_os_open(path, flags, *a, **k)
        for n in real:
            setattr(os, n, forbid(f"os.{n}"))
        os.open, subprocess.Popen, subprocess.run = guarded_os_open, forbid("Popen"), forbid("run")
        ctypes.CDLL = forbid("ctypes.CDLL")

        def restore():
            for n, f in real.items():
                setattr(os, n, f)
            os.open, subprocess.Popen, subprocess.run, ctypes.CDLL = real_os_open, real_popen, real_run, real_cdll
        self.addCleanup(restore)
        try:
            gpus, _ = self._machine()
            environ_file(self.proc, 77042, VK_ENV)
            map_fn, pid_fn = self._map_fn(), self.need("vulkan_map_for_pid")
            for _ in range(60):
                map_fn(dict(VK_ENV))
                pid_fn("77042")
                self.clock.sleep(1.0)
        except _Touched as e:
            self.fail(f"not read-only: {e}")
        self.assertEqual([g.backend_dev for g in gpus], ["Vulkan3", "Vulkan1", "Vulkan0", "Vulkan2"])
        # no code path runs vulkaninfo any more: the word survives in comments only (a string
        # literal or a docstring is code; a comment is not in the tree)
        with open(PROG) as f:
            tree = ast.parse(f.read())
        literals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        self.assertEqual([s for s in literals if "vulkaninfo" in s], [])


# ================================================================ I. round 7c: four decisions
# Fabrizio, 17/09: «punto 2 ok, accetto il tuo consiglio. 3a ok, … 3d ok. 4 va bene B come
# consigli.» Each item has its own class, its own RED run and its own verdict (p6).
#
# R1: freq1_input / freq2_input are never read on a legacy card. smu7_hwmgr.c: smu7_read_sensor
# GFX_SCLK / GFX_MCLK send PPSMC_MSG_API_GetSclkFrequency / GetMclkFrequency (4011-4012,
# 4018-4019) and smu7_print_clock_levels PP_SCLK / PP_MCLK send the SAME two messages (4978-4979,
# 4994-4995), starring the first DPM level at or above the returned clock (4981-4987, 4997-5003):
# the two hwmon clocks cost two messages per idle burst for a number the tables already carry.
class AmdNoHwmonClocks(_AmdRound7):
    THREE = ("pp_dpm_sclk", "pp_dpm_mclk", "power1_average")

    def _smc_opened(self):
        return [os.path.basename(p) for p in self._opened if os.path.basename(p) in self.SMC]

    # ---- AC-2 R1(a): one idle burst is exactly the two tables and the power
    def test_idle_burst_opens_exactly_the_two_tables_and_the_power(self):
        g = self._gpu()
        self._opened.clear()
        s = self._tick(g)
        self.assertEqual(sorted(self._smc_opened()), sorted(self.THREE))
        self.assertEqual((s["sclk"], s["mclk"]), (300, 1250))
        self.assertAlmostEqual(s["power"], 43.153, places=3)

    # ---- R1(b): a simulated minute idle, then a minute under load
    def test_freq_files_never_opened_over_a_minute(self):
        g = self._gpu()
        self._opened.clear()
        for _ in range(60):
            self._tick(g)
        n = self._smc_opened()
        self.assertEqual((n.count("freq1_input"), n.count("freq2_input")), (0, 0))
        self.assertEqual([n.count(f) for f in self.THREE], [6, 6, 6])   # 18 per idle minute, was 30
        # under load nothing opens: this half held before the change and must keep holding
        self._forbid = self.SMC
        for _ in range(60):
            self._set_busy(g, 100)
            self._tick(g)

    # ---- R1(c): no hw_sclk / hw_mclk in the sample, no hwmon cell in the panel, none in --probe
    def test_hwmon_clocks_leave_the_sample_the_panel_and_probe(self):
        g = self._gpu()
        s = self._tick(g)
        a = self._amd(s)
        self.assertNotIn("hw_sclk", a)
        self.assertNotIn("hw_mclk", a)
        self.assertEqual(set(s["slow_fields"]), {"sclk", "mclk", "power"})
        rows = L._gpu_rows(s, 200)
        clocks = _text(_group(rows, "clocks"))
        self.assertNotIn("hwmon", clocks)
        self.assertIn("300* 461", clocks)                 # the tables and their marker stay
        self.assertIn("150 1250*", clocks)
        self.assertIn("idle only", _text(rows))
        self.assertNotIn("hw_sclk", "\n".join(L._amd_probe_lines(s)))

    # ---- R1(d) negative control: a card with gpu_metrics keeps its per-tick reads, no freq either
    def test_navi_like_card_keeps_its_per_tick_reads_without_freq(self):
        g = self._gpu(gpu_metrics=True)
        self._opened.clear()
        for _ in range(60):
            self._set_busy(g, 100)
            s = self._tick(g)
        n = self._smc_opened()
        self.assertEqual((n.count("freq1_input"), n.count("freq2_input")), (0, 0))
        self.assertEqual([n.count(f) for f in self.THREE], [60, 60, 60])
        self.assertEqual((s["sclk"], s["mclk"]), (300, 1250))
        self.assertAlmostEqual(s["power"], 43.153, places=3)


# R2: llama_processes() reads fdinfo only for the descriptors that link into /dev/dri. MEASURED
# by the auditor (strace, round 7b phase B, idle llama-server on the W9170): 633 fdinfo reads of
# that one process in 45 s, fds 7, 8, 9 … 24 each tick until the first with drm-memory-vram. The
# complete fd table of the captured llama-bench (fixture drm-fdinfo-fixture/llama-bench/fd-links,
# 12:13) is below verbatim: 37 descriptors, five of them under /dev/dri, and of those only the two
# on renderD128 carry drm-* keys (the fdinfo of 10, 14, 18 has no drm-driver line at all: the
# NVIDIA render nodes are not served by drm_show_fdinfo, so those reads never enter a driver).
LLAMA_BENCH_FD_LINKS = {
    0: "/dev/null",
    1: "/home/licky/kbuild/bench/firepro-w9170-moe-dark-champion-20260917-1207/L3.2-8X3B-MOE-Dark-Champion-Inst-18.4B-uncen-ablit_D_AU-Q5_k_s.md",
    2: "/home/licky/kbuild/bench/firepro-w9170-moe-dark-champion-20260917-1207/L3.2-8X3B-MOE-Dark-Champion-Inst-18.4B-uncen-ablit_D_AU-Q5_k_s.stderr",
    3: "socket:[734259]", 4: "/dev/nvidiactl", 5: "/dev/nvidia-modeset", 6: "/dev/nvidia0",
    7: "/dev/nvidia0", 8: "socket:[734260]", 9: "/dev/nvidiactl", 10: "/dev/dri/renderD129",
    11: "/dev/nvidia1", 12: "/dev/nvidia1", 13: "/dev/nvidiactl", 14: "/dev/dri/renderD130",
    15: "/dev/nvidia2", 16: "/dev/nvidia2", 17: "/dev/nvidiactl", 18: "/dev/dri/renderD131",
    19: "/home/licky/.cache/nvidia/GLCache/8af0ee58c7ae0533784c946d7ab38072/764c558612da1990/224726ace850fe55.toc",
    20: "/home/licky/.cache/nvidia/GLCache/8af0ee58c7ae0533784c946d7ab38072/764c558612da1990/224726ace850fe55.bin",
    21: "/home/licky/.cache/nvidia/GLCache/8af0ee58c7ae0533784c946d7ab38072/ef674370256f6447/224726ace850fe55.toc",
    22: "/home/licky/.cache/nvidia/GLCache/8af0ee58c7ae0533784c946d7ab38072/ef674370256f6447/224726ace850fe55.bin",
    23: "/memfd:allocation fd (deleted)", 24: "/dev/dri/renderD128", 25: "/dev/dri/renderD128",
    26: "/mnt/disk2tb/Modelli/L3.2-8X3B-MOE-Dark-Champion-Inst-18.4B-uncen-ablit_D_AU-Q5_k_s.gguf",
    27: "/dev/nvidia1", 28: "pipe:[727929]", 29: "pipe:[727929]", 30: "/dev/nvidia1",
    31: "/dev/nvidia1", 32: "/dev/nvidia1"}
LLAMA_BENCH_DRM_FDINFO = {10: NV_FDINFO, 14: NV_FDINFO, 18: NV_FDINFO,
                          24: LLAMA_BENCH_FDINFO, 25: LLAMA_BENCH_FDINFO_25}
PLAIN_FDINFO = "pos:\t0\nflags:\t0100002\nmnt_id:\t28\nino:\t1234\n"
LLAMA_MODEL = "L3.2-8X3B-MOE-Dark-Champion-Inst-18.4B-uncen-ablit_D_AU-Q5_k_s"


def llama_proc_tree(root, pid, comm, links, fdinfo=None, rss_kib=12345678, cmdline=None):
    """<root>/<pid>/{comm, status, cmdline, fd/<n> -> target, fdinfo/<n>} for one process the
    way llama_processes() reads it: EVERY fd has a fdinfo file (a plain one unless `fdinfo`
    names its text), so a read of the wrong fd is visible as an open, not as an absence."""
    p = os.path.join(root, str(pid))
    os.makedirs(os.path.join(p, "fd"))
    os.makedirs(os.path.join(p, "fdinfo"))
    _w(p, "comm", comm)
    _w(p, "status", f"Name:\t{comm}\nVmRSS:\t{rss_kib:8d} kB\nCapPrm:\t0000000000000000")
    argv = cmdline if cmdline is not None else [comm, "-m", links.get(26, f"/x/{LLAMA_MODEL}.gguf")]
    with open(os.path.join(p, "cmdline"), "w") as f:
        f.write("\x00".join(argv) + "\x00")
    for fd, target in links.items():
        os.symlink(target, os.path.join(p, "fd", str(fd)))
        with open(os.path.join(p, "fdinfo", str(fd)), "w") as f:
            f.write((fdinfo or {}).get(fd, PLAIN_FDINFO))
    return p


class LlamaProcesses(_AmdRound7):
    """R2: the per-tick process panel reads fdinfo only for /dev/dri descriptors; the figures,
    gpu_accel, the unreadable path and the CPU-only row are what they were. /proc is this
    test's own tree, and any open under the real /proc fails the test, so a program that still
    walks the real one is caught at its first read, before any real fdinfo."""

    def setUp(self):
        super().setUp()
        inner = L.open

        def trip(path, mode="r", *a, **k):
            if str(path).startswith("/proc/"):
                raise _Touched(f"real {path}")
            return inner(path, mode, *a, **k)
        L.open = trip

    def _procs(self):
        try:
            return self.need("llama_processes")()
        except _Touched as e:
            self.fail(f"llama_processes opened {e}")

    def _bench(self, pid=77042):
        return llama_proc_tree(self.proc, pid, "llama-bench", LLAMA_BENCH_FD_LINKS, LLAMA_BENCH_DRM_FDINFO)

    def _fdinfo_fds(self):
        return [os.path.basename(p) for p in self._fdinfo_reads()]

    # ---- AC-2 R2(a)+(b): only the /dev/dri fds, in fd order, up to the first with the key; same figures
    def test_fdinfo_read_only_for_dri_descriptors_and_figures_unchanged(self):
        self._bench()
        self._opened.clear()
        out = self._procs()
        self.assertEqual(self._fdinfo_fds(), ["10", "14", "18", "24"])
        self.assertEqual(len(out), 1)
        v = out[0]
        self.assertEqual((v["pid"], v["name"], v["model"]), ("77042", "llama-bench", LLAMA_MODEL))
        self.assertEqual((v["vram"], v["gtt"]), (12235116 // 1024, 68416 // 1024))
        self.assertEqual((v["read"], v["gpu_accel"], v["vram_src"], v["error"], v["caps"]),
                         (True, True, None, None, []))
        self.assertEqual(v["rss"], 12345678 // 1024)
        self.assertIn("VRAM 11948M", _text(L._llama_proc_rows(out, 120)))

    # ---- R2(c): a CPU-only server never has its fdinfo read, and its row says so
    def test_cpu_only_process_reads_no_fdinfo(self):
        llama_proc_tree(self.proc, 4242, "llama-server",
                        {0: "/dev/null", 3: "socket:[1]", 26: f"/x/{LLAMA_MODEL}.gguf", 28: "pipe:[2]"})
        self._opened.clear()
        out = self._procs()
        self.assertEqual(self._fdinfo_reads(), [])
        v = out[0]
        self.assertEqual((v["read"], v["gpu_accel"], v["vram"], v["gtt"], v["error"]), (False, False, 0, 0, None))
        self.assertIn("CPU only (no VRAM)", _text(L._llama_proc_rows(out, 120)))

    # ---- R2(d): a process this user may not read is handled as before
    @unittest.skipIf(os.geteuid() == 0, "root reads every fd directory")
    def test_unreadable_process_is_reported_not_fatal(self):
        p = self._bench(pid=99)
        for d in ("fd", "fdinfo"):                 # the kernel denies both to the same callers
            os.chmod(os.path.join(p, d), 0)
            self.addCleanup(os.chmod, os.path.join(p, d), 0o755)
        self._opened.clear()
        out = self._procs()
        self.assertEqual(self._fdinfo_reads(), [])
        v = out[0]
        self.assertEqual((v["read"], v["error"], v["caps"], v["gpu_accel"]), (False, "oserror", [], False))
        self.assertIn("owned by another user", _text(L._llama_proc_rows(out, 120)))

    # ---- negative control: an NVIDIA-only process still has its render-node fd read (no drm keys)
    def test_nvidia_only_process_reads_its_render_node_and_stays_unreported(self):
        llama_proc_tree(self.proc, 4343, "llama-server",
                        {0: "/dev/null", 4: "/dev/nvidiactl", 6: "/dev/nvidia0", 10: "/dev/dri/renderD129",
                         26: f"/x/{LLAMA_MODEL}.gguf"}, {10: NV_FDINFO})
        self._opened.clear()
        out = self._procs()
        self.assertEqual(self._fdinfo_fds(), ["10"])
        v = out[0]
        self.assertEqual((v["read"], v["gpu_accel"], v["error"]), (False, True, None))
        self.assertIn("not reported by driver (NVIDIA?)", _text(L._llama_proc_rows(out, 120)))

    # ---- R2(e): the budget over a simulated minute, per tick
    def test_budget_over_a_minute(self):
        self._bench()
        llama_proc_tree(self.proc, 4242, "llama-server",
                        {0: "/dev/null", 26: f"/x/{LLAMA_MODEL}.gguf"}, rss_kib=1024)
        self._opened.clear()
        for _ in range(60):
            out = self._procs()
            self.clock.sleep(1.0)
        fds = self._fdinfo_fds()
        self.assertEqual(len(fds), 60 * 4)                      # was 25 per tick for this table (fds 0–24)
        self.assertEqual(set(fds), {"10", "14", "18", "24"})
        self.assertEqual([v["pid"] for v in out], ["77042", "4242"])   # largest RSS first, as before


# R3: discovery runs no program. gpu_product_name() ran `lspci -mm -s <addr>` once per card (four
# runs at every start here). The auditor's fixture (pci-names-fixture, 14:59, GPUs idle) shows
# lspci printed exactly the pci.ids device string for all four cards, and that
# /usr/share/hwdata/pci.ids is a symlink to /usr/share/misc/pci.ids (1.4 MB). The four rows below
# are the fixture's (address, vendor, device, lspci -mm device string), verbatim.
PCI_NAMES_FIXTURE = (
    ("0000:23:00.0", "0x1002", "0x67a0", "Hawaii XT GL [FirePro W9100]"),
    ("0000:12:00.0", "0x10de", "0x15f8", "GP100GL [Tesla P100 PCIe 16GB]"),
    ("0000:14:00.0", "0x10de", "0x1f08", "TU106 [GeForce RTX 2060 Rev. A]"),
    ("0000:1b:00.0", "0x10de", "0x15f8", "GP100GL [Tesla P100 PCIe 16GB]"),
)
# a pci.ids in the real file's grammar: comment header, vendor lines, device lines (one tab),
# subsystem lines (two tabs, never a device), and the class list that closes the file
PCI_IDS_TEXT = """#
#\tList of PCI ID's
#
# Syntax:
# vendor  vendor_name
#\tdevice  device_name\t\t\t\t<-- single tab
#\t\tsubvendor subdevice  subsystem_name\t<-- two tabs

0e11  Compaq Computer Corporation
\t0001  PCI to EISA Bridge
1002  Advanced Micro Devices, Inc. [AMD/ATI]
\t6798  Tahiti XT [Radeon HD 7970/8970 OEM / R9 280X]
\t67a0  Hawaii XT GL [FirePro W9100]
\t\t1002 0b00  FirePro S9170
\t67b0  Hawaii XT / Grenada XT [Radeon R9 290X/390X]
10de  NVIDIA Corporation
\t15f8  GP100GL [Tesla P100 PCIe 16GB]
\t1f08  TU106 [GeForce RTX 2060 Rev. A]
\t\t1458 37c2  RTX 2060
\t1f09  TU106 [GeForce GTX 1660 SUPER]
8086  Intel Corporation
\t56a0  DG2 [Arc A770]

# List of known device classes, subclasses and programming interfaces

# Syntax:
# C class\tclass_name
#\tsubclass\tsubclass_name \t\t<-- single tab
#\t\tprog-if  prog-if_name \t<-- two tabs

C 00  Unclassified device
\t00  Non-VGA unclassified device
C 03  Display controller
\t00  VGA compatible controller
\t\t00  VGA controller
\t02  3D controller
"""


def forbid_programs(tc):
    """For the rest of the test, every way this process could start another program raises
    _Touched: subprocess.run/Popen and the os.exec*/posix_spawn*/system/popen/fork family. A
    tripwire and not a PATH trick: a program that PATH cannot find is still a program the code
    tried to run. Restored on cleanup, in LIFO order with any forbid a test installed itself."""
    import subprocess

    def forbid(name):
        def f(*a, **k):
            raise _Touched(name)
        return f
    names = ("execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe",
             "posix_spawn", "posix_spawnp", "system", "popen", "fork")
    real = {n: getattr(os, n) for n in names if hasattr(os, n)}
    real_popen, real_run = subprocess.Popen, subprocess.run

    def restore():
        for n, f in real.items():
            setattr(os, n, f)
        subprocess.Popen, subprocess.run = real_popen, real_run
    tc.addCleanup(restore)
    for n in real:
        setattr(os, n, forbid(f"os.{n}"))
    subprocess.Popen, subprocess.run = forbid("Popen"), forbid("run")


class PciNames(_Base):
    """R3: product names from pci.ids read once, the vendor fallback kept, no program at all."""

    def setUp(self):
        super().setUp()
        self._opened = []
        real_open = open

        def trip(path, *a, **k):
            self._opened.append(str(path))
            return real_open(path, *a, **k)
        L.open = trip
        self.addCleanup(lambda: delattr(L, "open") if "open" in vars(L) else None)
        self.ids = os.path.join(self.tmp.name, "pci.ids")
        with open(self.ids, "w") as f:
            f.write(PCI_IDS_TEXT)
        # the first path is absent on purpose: the reader takes the first that exists
        self._paths(os.path.join(self.tmp.name, "absent", "pci.ids"), self.ids)
        cache = self.need("_pci_ids_cache")
        saved = dict(cache)
        cache.clear()
        self.addCleanup(lambda: (cache.clear(), cache.update(saved)))
        forbid_programs(self)

    def _paths(self, *paths):
        saved = L._PCI_IDS_PATHS
        L._PCI_IDS_PATHS = tuple(paths)
        self.addCleanup(setattr, L, "_PCI_IDS_PATHS", saved)

    def _dev(self, addr, vendor, device=None):
        d = os.path.join(self.tmp.name, "pci", addr)
        _w(d, "vendor", vendor)
        if device is not None:
            _w(d, "device", device)
        return d

    def _name(self, dev, vendor):
        fn = self.need("gpu_product_name")
        self.assertEqual(list(inspect.signature(fn).parameters), ["dev_path", "vendor_id"],
                         "behaviour missing: gpu_product_name(dev_path, vendor_id), no address for lspci")
        try:
            return fn(dev, vendor)
        except _Touched as e:
            self.fail(f"naming executed a program: {e}")

    # ---- AC-2 R3(b)+(c): the fixture names, byte-identical to lspci's, from one read of the database
    def test_names_are_the_fixture_lspci_strings_from_one_read(self):
        for addr, vendor, device, expected in PCI_NAMES_FIXTURE:
            self.assertEqual(self._name(self._dev(addr, vendor, device), vendor), expected, addr)
        self.assertEqual(self._opened.count(self.ids), 1, "pci.ids is static: read once for four cards")
        self.assertEqual(self._name(self._dev("0000:00:00.0", "0x8086", "0x56a0"), "0x8086"), "DG2 [Arc A770]")
        self.assertEqual(self._opened.count(self.ids), 1)

    # ---- R3(a): no lspci in the program, no program executed while naming
    def test_no_lspci_and_no_program(self):
        import ast
        self.assertFalse(hasattr(L, "_name_from_lspci"), "behaviour missing: _name_from_lspci removed")
        with open(PROG) as f:
            tree = ast.parse(f.read())
        self.assertEqual([n.value for n in ast.walk(tree)
                          if isinstance(n, ast.Constant) and n.value == "lspci"], [])
        for name in ("gpu_product_name", "_name_from_pci_ids"):
            src = inspect.getsource(self.need(name))
            self.assertNotIn("subprocess", src, name)
            self.assertNotIn("which(", src, name)
        # the tripwire is live: a name resolved with programs forbidden proves the path runs none
        self.assertEqual(self._name(self._dev("0000:23:00.0", "0x1002", "0x67a0"), "0x1002"),
                         "Hawaii XT GL [FirePro W9100]")

    # ---- R3(d): no database, no entry, unknown vendor, no device file → the vendor fallback
    def test_vendor_fallback_when_nothing_answers(self):
        self.assertEqual(self._name(self._dev("0000:12:00.0", "0x10de", "0xffff"), "0x10de"), "NVIDIA GPU")
        self.assertEqual(self._name(self._dev("0000:0a:00.0", "0x1af4", "0x1050"), "0x1af4"), "Unknown GPU")
        self.assertEqual(self._name(self._dev("0000:23:00.0", "0x1002"), "0x1002"), "AMD GPU")
        self.assertEqual(self._name(self._dev("0000:23:00.0", "0x1002", ""), "0x1002"), "AMD GPU")
        # a subsystem line is never a device: 0b00 under 1002 is the S9170's subsystem id
        self.assertEqual(self._name(self._dev("0000:23:00.1", "0x1002", "0x0b00"), "0x1002"), "AMD GPU")
        L._pci_ids_cache.clear()
        self._paths(os.path.join(self.tmp.name, "absent", "pci.ids"))
        self.assertEqual(self._name(self._dev("0000:12:00.0", "0x10de", "0x15f8"), "0x10de"), "NVIDIA GPU")
        self.assertEqual(self._name(self._dev("0000:23:00.0", "0x1002", "0x67a0"), "0x1002"), "AMD GPU")

    # ---- the --help line lists the helpers that can still run
    def test_help_names_no_lspci(self):
        import contextlib
        import io
        import sys
        saved = sys.argv
        sys.argv = ["llamagputop.py", "--help"]
        self.addCleanup(setattr, sys, "argv", saved)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            L.main()
        text = buf.getvalue()
        self.assertIn("optional helpers used when present: intel_gpu_top, nvtop, nvidia-smi.", text)
        self.assertNotIn("lspci", text)


# R4: the NVIDIA GPU temperature turns CRIT where the card is already cutting its clocks.
# nvidia-smi -q -x reports per card gpu_temp_slow_threshold (slowdown), gpu_temp_max_threshold
# (shutdown) and, on the 2060, gpu_temp_max_gpu_threshold (max operating). Fabrizio, 17/09: «4 va
# bene B come consigli»: CRIT from the slowdown point (else shutdown, as before), WARN from max
# operating when it lies below the CRIT point, else the fixed 80 °C. Proven the way the CHANGELOG
# did: every integer 60–95 °C through the real colour function on the real sample() output. The
# AMD hwmon crit/emergency mapping is not part of the decision and must not move.
class NvidiaTempBands(_Base):
    def _bands(self, thresholds):
        return {t: L._temp_attr_th(t, thresholds) for t in range(60, 96)}

    def _expect(self, warn_from, crit_from):
        return {t: (L.CRIT if t >= crit_from else L.WARN if t >= warn_from else L.OK) for t in range(60, 96)}

    def _sample(self, bus, temp, detail=None):
        detail = dict(_gpu(self, bus)) if detail is None else detail
        feed = _FakeFeed({0: _core(temp)}, {bus: 0}, {bus: detail}, {}, time.monotonic())
        g = L.NvidiaGpu("x", 0, feed)
        g.pci_addr = bus
        s = g.sample()
        self.assertIn("temp_crit", s, "behaviour missing: NVIDIA temp_crit")
        return s

    def _attr_at(self, bus, temp, detail=None):
        s = self._sample(bus, temp, detail)
        attrs = _attrs_of(L._gpu_rows(s, 140), f"{float(temp)}°C")
        self.assertEqual(len(attrs), 1, attrs)
        return attrs[0]

    # ---- AC-2 R4(a): RTX 2060 (max op 88, slowdown 90, shutdown 93): OK < 80, WARN 80–89, CRIT ≥ 90
    def test_2060_bands(self):
        s = self._sample("0000:14:00.0", 50.0)
        self.assertEqual(s["temp_crit"], {"gpu": (88, 90)})
        self.assertEqual(self._bands(s["temp_crit"]["gpu"]), self._expect(80, 90))
        self.assertEqual([self._attr_at("0000:14:00.0", t) for t in (79.0, 80.0, 89.0, 90.0)],
                         [L.OK, L.WARN, L.WARN, L.CRIT])
        # the limits row still prints the driver's own numbers, unchanged
        text = _text(L._gpu_rows(s, 140))
        for needle in ("slowdown 90", "shutdown 93", "max op 88"):
            self.assertIn(needle, text)

    # ---- R4(b): Tesla P100 (slowdown 82, shutdown 85, no max op): OK < 80, WARN 80–81, CRIT ≥ 82
    def test_p100_bands(self):
        s = self._sample("0000:12:00.0", 50.0)
        self.assertEqual(s["temp_crit"], {"gpu": (0, 82)})
        self.assertEqual(self._bands(s["temp_crit"]["gpu"]), self._expect(80, 82))
        self.assertEqual([self._attr_at("0000:12:00.0", t) for t in (79.0, 80.0, 81.0, 82.0)],
                         [L.OK, L.WARN, L.WARN, L.CRIT])
        self.assertIn("slowdown 82 · shutdown 85", _text(L._gpu_rows(s, 140)))

    # ---- R4(c): a card reporting only the shutdown point keeps CRIT there; only max op keeps its old shape
    def test_shutdown_only_and_max_op_only_cards(self):
        detail = {k: v for k, v in _gpu(self, "0000:14:00.0").items() if k not in ("temp_slowdown", "temp_max_op")}
        s = self._sample("0000:14:00.0", 50.0, detail)
        self.assertEqual(s["temp_crit"], {"gpu": (0, 93)})
        # below the driver's point _temp_attr_th falls back to the fixed 80/90 °C, so a
        # shutdown point above 90 is reached after the fixed CRIT, as before round 7c
        self.assertEqual(self._bands(s["temp_crit"]["gpu"]), self._expect(80, 90))
        s = self._sample("0000:14:00.0", 50.0, dict(detail, temp_shutdown=85))
        self.assertEqual(s["temp_crit"], {"gpu": (0, 85)})
        self.assertEqual(self._bands(s["temp_crit"]["gpu"]), self._expect(80, 85))
        # neither slowdown nor shutdown: the rule names no CRIT point, and the pre-round-7c
        # shape (WARN from max op, no driver CRIT) is kept rather than replaced by a guess
        detail = {k: v for k, v in _gpu(self, "0000:14:00.0").items() if k not in ("temp_slowdown", "temp_shutdown")}
        s = self._sample("0000:14:00.0", 50.0, detail)
        self.assertEqual(s["temp_crit"], {"gpu": (88, 0)})
        # and a card with no threshold at all carries no temp_crit
        detail = {k: v for k, v in _gpu(self, "0000:14:00.0").items()
                  if k not in ("temp_slowdown", "temp_shutdown", "temp_max_op")}
        feed = _FakeFeed({0: _core(50.0)}, {"0000:14:00.0": 0}, {"0000:14:00.0": detail}, {}, time.monotonic())
        g = L.NvidiaGpu("x", 0, feed)
        g.pci_addr = "0000:14:00.0"
        self.assertNotIn("temp_crit", g.sample())

    # ---- R4(d) negative control: AMD colouring is what it was (hwmon crit / emergency)
    def test_amd_colouring_unchanged(self):
        g = L.AmdGpu(w9170_tree(self.tmp.name), "FirePro W9170")
        self.assertEqual(g.temp_crit, {"edge": (104, 0)})
        self.assertEqual(self._bands(g.temp_crit["edge"]), self._expect(80, 90))
        self.assertEqual(self._bands((100, 105)), {t: (L.CRIT if t >= 90 else L.WARN if t >= 80 else L.OK)
                                                   for t in range(60, 96)})
        self.assertEqual(self._bands((85, 95)), {t: (L.CRIT if t >= 95 else L.WARN if t >= 80 else L.OK)
                                                 for t in range(60, 96)})

    @needs_baseline
    def test_amd_colouring_matches_the_baseline(self):
        for th in ((104, 0), (100, 105), (85, 95), (110, 115), None):
            self.assertEqual(self._bands(th), {t: B._temp_attr_th(t, th) for t in range(60, 96)}, th)


# ================================================================ round 8a: the NVIDIA feed
# on persistent nvidia-smi processes (V3; Fabrizio, 17/09: «vai con V3»). MEASURED by the auditor
# (five 300 s runs, N = 150 two-second windows, Mann-Whitney against the first V0 run): the feed
# of rounds 1–7c spawned 136 nvidia-smi per minute for 4.79 s of nvidia-smi CPU per minute;
# three persistent processes (core `-lms 1000`, compute-apps `-l 1`, `-q -x -l 3`) plus the
# index/bus table once spawned 0.8 per minute for 2.09 s (−56 %, z −12.4, p 3·10⁻³⁵), with the
# same fields and the same parser. Nothing here runs a real nvidia-smi: the fake-factory tests
# hand the feed pipe-backed fakes under forbid_programs; the children tests put a fake
# `nvidia-smi` alone on PATH and refuse anything else before it could start.
_NV_CORE_ROWS = ("0, Tesla P100-PCIE-16GB, 0, 0, 5, 16384, 38, 25.53, 250.00, 405, 715\n"
                 "1, NVIDIA GeForce RTX 2060, 10, 5, 171, 6144, 32, 2.06, 190.00, 300, 405\n")
_NV_INDEX_ROWS = "0, 00000000:12:00.0\n1, 00000000:14:00.0\n"
_NV_APPS_ROWS = "3802, 122\n"          # apps.csv: one line per compute process per iteration


def _nv_kind(args):
    """Which of the feed's four nvidia-smi invocations an argv is, told by its own flags."""
    a = list(args)
    if "-x" in a:
        return "detail"
    if any(x.startswith("--query-compute-apps") for x in a):
        return "apps"
    if "--query-gpu=index,pci.bus_id" in a:
        return "index"
    if any(x.startswith("--query-gpu=") for x in a):
        return "core"
    return "?"


class _FakeChild:
    """A Popen look-alike on a real pipe. The test writes what nvidia-smi would print and closes
    the pipe to make the child exit; terminate/kill/wait/poll record what the feed did to it.
    poll() reaps like waitpid does: a child whose exit was seen counts as reaped."""
    def __init__(self, args):
        self.args = list(args)
        r, w = os.pipe()
        self.stdout = os.fdopen(r, "rb", buffering=0)
        self._w = w
        self.returncode = None
        self.signals = []
        self.waited = False
        self.ignores_term = False
        self.born = None

    def write(self, text):
        os.write(self._w, text.encode() if isinstance(text, str) else text)

    def exit(self, rc=0):
        if self._w is not None:
            os.close(self._w)
            self._w = None
        if self.returncode is None:
            self.returncode = rc

    def poll(self):
        if self.returncode is not None:
            self.waited = True
        return self.returncode

    def terminate(self):
        self.signals.append("TERM")
        if not self.ignores_term:
            self.exit(-15)

    def kill(self):
        self.signals.append("KILL")
        self.exit(-9)

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired(self.args, timeout)
        self.waited = True
        return self.returncode

    def close(self):
        self.exit()
        self.stdout.close()


class _FakeSpawner:
    """The feed's process factory under the test's control: every launch is recorded with the
    fake clock's time; `dead_at_birth` children exit at once, `fail` refuses to launch at all."""
    def __init__(self, clock, dead_at_birth=False, fail=False):
        self.clock = clock
        self.procs = []
        self.attempts = []
        self.dead_at_birth = dead_at_birth
        self.fail = fail

    def __call__(self, args):
        self.attempts.append((self.clock.now, _nv_kind(args)))
        if self.fail:
            raise OSError(2, "nvidia-smi vanished")
        p = _FakeChild(args)
        p.born = self.clock.now
        self.procs.append(p)
        if self.dead_at_birth:
            p.exit(1)
        return p

    def of(self, kind):
        return [p for p in self.procs if _nv_kind(p.args) == kind]

    def close(self):
        for p in self.procs:
            p.close()


class NvidiaSmiSplit(_Base):
    """The stream splitter alone: bytes in, whole records out, a partial tail waits."""
    def _stream(self, xml=False):
        return self.need("_NvSmiStream")(["nvidia-smi"], persistent=True, xml=xml)

    def test_lines_split_on_newline_and_the_tail_waits(self):
        s = self._stream()
        self.assertEqual(s.records(b"0, a\n1, b\n2, c"), ["0, a", "1, b"])
        self.assertEqual(s.records(b"c\n"), ["2, cc"])
        self.assertEqual(s.records(b""), [])
        self.assertEqual(s.records(b"\n"), [""])

    def test_two_documents_in_one_chunk(self):
        s = self._stream(xml=True)
        docs = s.records((NVQ_XML + NVQ_XML).encode())
        self.assertEqual(len(docs), 2)
        for d in docs:
            self.assertTrue(d.startswith("<?xml"), d[:40])
            self.assertTrue(d.endswith("</nvidia_smi_log>"), d[-40:])
        # the newline between documents is not carried into the next one
        third = s.records(NVQ_XML.encode())
        self.assertEqual(len(third), 1)
        self.assertTrue(third[0].startswith("<?xml"))
        self.assertEqual(L.parse_nvidia_smi_xml(third[0])["gpus"], L.parse_nvidia_smi_xml(NVQ_XML)["gpus"])

    def test_one_document_across_chunks_and_a_truncated_one_is_never_returned(self):
        s = self._stream(xml=True)
        raw = NVQ_XML.encode()
        cut1, cut2 = len(raw) // 3, 2 * len(raw) // 3
        self.assertEqual(s.records(raw[:cut1]), [])
        self.assertEqual(s.records(raw[cut1:cut2]), [])
        docs = s.records(raw[cut2:])
        self.assertEqual(docs, [NVQ_XML.strip()])
        # a chunk boundary inside the closing tag itself
        end = raw.rfind(b"</nvidia_smi_log>")
        self.assertEqual(s.records(raw[:end + 5]), [])
        self.assertEqual(len(s.records(raw[end + 5:])), 1)
        # a document that never closes is never a record, however long it waits
        self.assertEqual(s.records(raw[:cut2]), [])
        self.assertEqual(s.records(b""), [])

    def test_buffer_cap_both_sides(self):
        cap = self.need("_NV_BUFFER_MAX")
        # positive: a stream that never closes a document is dropped past the cap, and the next
        # whole document still comes through
        s = self._stream(xml=True)
        self.assertEqual(s.records(b"<nvidia_smi_log>" + b"x" * cap), [])
        self.assertEqual(s.records(NVQ_XML.encode()), [NVQ_XML.strip()])
        # negative: loop.xml, three whole documents for three cards, is 232172 bytes, far below
        # the cap; three documents in one chunk all come through
        self.assertLess(232172, cap)
        self.assertEqual(len(self._stream(xml=True).records((NVQ_XML * 3).encode())), 3)


class NvidiaPersistentFeed(_Base):
    """The feed driven by hand: fake children on real pipes, a fake clock, no program."""
    def setUp(self):
        super().setUp()
        forbid_programs(self)
        self.clock = _FakeClock()
        real_time = L.time
        L.time = self.clock
        self.addCleanup(setattr, L, "time", real_time)
        saved = dict(L._NV_PROC_MEM)
        L._NV_PROC_MEM.clear()
        self.addCleanup(lambda: (L._NV_PROC_MEM.clear(), L._NV_PROC_MEM.update(saved)))
        self.spawner = _FakeSpawner(self.clock)
        self.addCleanup(self.spawner.close)

    def _feed(self, spawner=None, tool_ok=True):
        f = L._NvidiaFeed()
        f.tool_ok = tool_ok
        f.spawn = spawner or self.spawner
        self.assertIsNotNone(getattr(f, "_service", None), "behaviour missing: _NvidiaFeed._service")
        return f

    def _step(self, f, seconds=0.0, n=1):
        """Advance the fake clock and run one service pass; the pass waits 0 real seconds."""
        for _ in range(n):
            self.clock.now += seconds
            f._service(0)

    def _die(self, f, child):
        """Close the child's pipe and run the pass that sees it; returns when the feed saw it."""
        child.exit(1)
        self._step(f, 0.5)
        return self.clock.now

    # ---- (a) spawns
    def test_start_spawns_three_loops_and_one_index_then_none_for_a_minute(self):
        f = self._feed()
        self._step(f)
        self.assertEqual(sorted(_nv_kind(p.args) for p in self.spawner.procs),
                         ["apps", "core", "detail", "index"])
        self.assertEqual(self.spawner.of("core")[0].args,
                         ["nvidia-smi", "--query-gpu=" + ",".join(L._NVSMI_FIELDS),
                          "--format=csv,noheader,nounits", "-lms", "1000"])
        self.assertEqual(self.spawner.of("apps")[0].args,
                         ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                          "--format=csv,noheader,nounits", "-l", "1"])
        self.assertEqual(self.spawner.of("detail")[0].args, ["nvidia-smi", "-q", "-x", "-l", "3"])
        self.assertEqual(self.spawner.of("index")[0].args,
                         ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"])
        self.spawner.of("index")[0].exit(0)          # a one-shot answers and exits
        self._step(f, 1.0, 60)                       # a simulated minute with the three loops alive
        self.assertEqual(len(self.spawner.procs), 4)
        self.assertEqual([p.signals for p in self.spawner.procs], [[], [], [], []])

    # ---- (b) the core stream
    def test_core_lines_update_by_index_with_fresh_ts_and_the_old_shape(self):
        f = self._feed()
        self._step(f)
        core = self.spawner.of("core")[0]
        core.write("0, Tesla P100-PCIE-16GB, 0, 0, 5, 16384, 38, 25.53, 250.00, 405, 715\n")
        self._step(f, 2.0)
        self.assertEqual(f.by_index, {0: {"name": "Tesla P100-PCIE-16GB", "util": 0.0, "mem_util": 0.0,
                                          "vram_used": 5.0, "vram_total": 16384.0, "temp_main": 38.0,
                                          "power": 25.53, "power_cap": 250.0, "sclk": 405.0,
                                          "mclk": 715.0, "ts": self.clock.now}})
        # a row that arrives in two pieces lands once, whole, stamped when it completed
        core.write("1, NVIDIA GeForce RTX 2060, 10, 5, 171, 6144, 32, 2.06, 190.00, 300, 4")
        self._step(f, 1.0)
        self.assertNotIn(1, f.by_index)
        core.write("05\n")
        self._step(f, 1.0)
        self.assertEqual((f.by_index[1]["mclk"], f.by_index[1]["ts"]), (405.0, self.clock.now))
        # a placeholder field stays None, as today's num() made it
        core.write("0, Tesla P100-PCIE-16GB, [N/A], 0, 5, 16384, 38, 25.53, 250.00, 405, 715\n")
        self._step(f, 1.0)
        self.assertIsNone(f.by_index[0]["util"])
        self.assertEqual(f.by_index[0]["ts"], self.clock.now)

    def test_index_table_lands_at_eof_and_is_reread_only_on_a_core_restart(self):
        f = self._feed()
        self._step(f)
        idx = self.spawner.of("index")[0]
        idx.write(_NV_INDEX_ROWS)
        self._step(f)
        self.assertEqual(f.index_by_bus, {})          # a table is one whole answer: taken at exit
        idx.exit(0)
        self._step(f)
        self.assertEqual(f.index_by_bus, {"0000:12:00.0": 0, "0000:14:00.0": 1})
        self._step(f, 1.0, 30)
        self.assertEqual(len(self.spawner.of("index")), 1)
        # the apps loop dies: it alone is restarted, the static table is not re-read
        self._die(f, self.spawner.of("apps")[0])
        self._step(f, 1.0, 5)
        self.assertEqual((len(self.spawner.of("apps")), len(self.spawner.of("index"))), (2, 1))
        # the core loop dies: its restart re-reads the table its rows are joined with
        self._die(f, self.spawner.of("core")[0])
        self._step(f, 1.0, 5)
        self.assertEqual((len(self.spawner.of("core")), len(self.spawner.of("index"))), (2, 2))
        # an empty answer never replaces a table (today: `if by_bus`)
        self.spawner.of("index")[1].exit(0)
        self._step(f)
        self.assertEqual(f.index_by_bus, {"0000:12:00.0": 0, "0000:14:00.0": 1})

    # ---- (c) the detail stream
    def test_detail_documents_split_on_the_boundary_and_a_truncated_last_is_never_parsed(self):
        f = self._feed()
        self._step(f)
        parsed = []
        real = L.parse_nvidia_smi_xml
        L.parse_nvidia_smi_xml = lambda text: (parsed.append(text), real(text))[1]
        self.addCleanup(setattr, L, "parse_nvidia_smi_xml", real)
        det = self.spawner.of("detail")[0]
        # two documents in one chunk, as `-l 3` prints them back to back
        det.write(NVQ_XML + NVQ_XML)
        self._step(f, 3.0)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(f.detail_by_bus, real(NVQ_XML)["gpus"])
        self.assertEqual(f.driver, {"driver_version": "580.173.02", "cuda_version": "13.0"})
        self.assertEqual(f.detail_ts, self.clock.now)
        # one document across two chunks: nothing until the boundary arrives
        cut = NVQ_XML.index('<gpu id="00000000:14:00.0">')
        det.write(NVQ_XML[:cut])
        self._step(f, 3.0)
        self.assertEqual(len(parsed), 2)
        self.assertNotEqual(f.detail_ts, self.clock.now)
        det.write(NVQ_XML[cut:])
        self._step(f, 3.0)
        self.assertEqual((len(parsed), f.detail_ts), (3, self.clock.now))
        # the pipe closes mid-document: the truncated tail never reaches the parser
        det.write(NVQ_XML[:cut])
        det.exit(0)
        self._step(f, 3.0)
        self.assertEqual(len(parsed), 3)
        for doc in parsed:
            self.assertTrue(doc.startswith("<?xml") and doc.endswith("</nvidia_smi_log>"))

    def test_garbage_records_are_skipped_and_the_streams_go_on(self):
        f = self._feed()
        self._step(f)
        self.spawner.of("core")[0].write("not, a, row\nx, y, z, w, v, u, t, s, r, q, p\n" + _NV_CORE_ROWS)
        self.spawner.of("detail")[0].write("<nvidia_smi_log><broken</nvidia_smi_log>" + NVQ_XML)
        self.spawner.of("apps")[0].write("3802\nabc, def\n" + _NV_APPS_ROWS)
        self._step(f)
        self.assertEqual(sorted(f.by_index), [0, 1])
        self.assertEqual(f.detail_by_bus, L.parse_nvidia_smi_xml(NVQ_XML)["gpus"])
        self.assertEqual(L._NV_PROC_MEM, {"3802": (122, self.clock.now)})
        self.assertEqual(len(self.spawner.procs), 4)

    # ---- (d) compute-apps silence
    def test_apps_silence_keeps_fresh_entries_ages_old_ones_and_is_not_a_death(self):
        f = self._feed()
        self._step(f)
        apps = self.spawner.of("apps")[0]
        apps.write(_NV_APPS_ROWS)
        self._step(f)
        t0 = self.clock.now
        self.assertEqual(L._NV_PROC_MEM, {"3802": (122, t0)})
        # thirty silent seconds: kept with its own stamp, and nothing is restarted
        self._step(f, 1.0, 30)
        self.assertEqual(L._NV_PROC_MEM, {"3802": (122, t0)})
        self.assertEqual((len(self.spawner.procs), apps.signals), (4, []))
        # past 60 s it is dropped although no apps line ever arrived to trigger the sweep
        self._step(f, 1.0, 31)
        self.assertEqual(L._NV_PROC_MEM, {})
        self.assertEqual(len(self.spawner.procs), 4)
        # the consumer's own 6 s rule is untouched: a fresh entry is used, an old one is not
        apps.write(_NV_APPS_ROWS)
        self._step(f)
        v = {"pid": "3802", "read": False, "gpu_accel": True, "vram": 0, "gtt": 0, "vram_src": None}
        self.assertEqual(L._nv_proc_fill(dict(v))["vram_src"], "nvidia-smi")
        self.clock.now += 6.5
        self.assertIsNone(L._nv_proc_fill(dict(v))["vram_src"])

    # ---- (e) staleness on the existing guards
    def test_stalled_streams_make_the_sample_stale_on_the_existing_guards(self):
        f = self._feed()
        self._step(f)
        self.spawner.of("core")[0].write(_NV_CORE_ROWS)
        self.spawner.of("detail")[0].write(NVQ_XML)
        self._step(f)
        g = L.NvidiaGpu("x", 1, f)
        g.pci_addr = "0000:14:00.0"
        s = g.sample()
        self.assertEqual((s.get("util"), s.get("pstate"), s.get("nv_driver")), (10.0, "P8", "580.173.02"))
        self.assertNotIn("needs", s)
        # both loops alive and silent: the core rows age out at 6 s, the detail at _SANE_TTL
        self._step(f, 1.0, 7)
        s = g.sample()
        self.assertEqual(s.get("needs"), {k: "nvidia-smi returned nothing for this card"
                                          for k in ("util", "vram", "temp", "power", "clocks")})
        self.assertEqual(s.get("pstate"), "P8")
        self._step(f, 1.0, 4)
        s = g.sample()
        self.assertNotIn("pstate", s)
        self.assertNotIn("throttle", s)
        self.assertEqual(len(self.spawner.procs), 4)          # silence is not death

    # ---- (f) restart with backoff
    def test_crash_loop_is_bounded_by_backoff(self):
        sp = _FakeSpawner(self.clock, dead_at_birth=True)
        self.addCleanup(sp.close)
        f = self._feed(sp)
        t0 = self.clock.now
        self._step(f)
        self._step(f, 0.5, 130)                                # to t0 + 65, two passes a second
        for kind in ("core", "apps", "detail", "index"):
            self.assertEqual([p.born - t0 for p in sp.of(kind)], [0.0, 1.0, 3.0, 7.0, 15.0, 31.0, 61.0], kind)
        self.assertEqual(len([p for p in sp.procs if p.born - t0 < 60]), 24)
        # steady state: one launch per 30 s per stream, 8 per minute for the four (136 today)
        self._step(f, 0.5, 240)                                # to t0 + 185
        self.assertEqual(len([p for p in sp.procs if 65 <= p.born - t0 < 185]), 16)

    def test_spawn_failure_follows_the_same_backoff(self):
        sp = _FakeSpawner(self.clock, fail=True)
        f = self._feed(sp)
        t0 = self.clock.now
        self._step(f)
        self._step(f, 0.5, 130)
        self.assertEqual([t - t0 for t, k in sp.attempts if k == "core"], [0.0, 1.0, 3.0, 7.0, 15.0, 31.0, 61.0])
        self.assertEqual(len([t for t, _k in sp.attempts if t - t0 < 60]), 24)
        self.assertEqual(sp.procs, [])

    def test_a_child_that_lived_a_minute_restarts_on_the_first_delay(self):
        f = self._feed()
        self._step(f)
        seen = self._die(f, self.spawner.of("core")[0])
        self._step(f, 0.5, 2)
        self.assertEqual([p.born for p in self.spawner.of("core")][1:], [seen + 1.0])
        seen = self._die(f, self.spawner.of("core")[1])
        self._step(f, 0.5, 3)                                   # +1.5 s: the second delay is 2 s
        self.assertEqual(len(self.spawner.of("core")), 2)
        self._step(f, 0.5)
        self.assertEqual([p.born for p in self.spawner.of("core")][2:], [seen + 2.0])
        # negative control: a child that lived a minute is not a crash loop
        self._step(f, 1.0, 61)
        seen = self._die(f, self.spawner.of("core")[2])
        self._step(f, 0.5, 2)
        self.assertEqual([p.born for p in self.spawner.of("core")][3:], [seen + 1.0])
        # and a living child is never restarted
        self.assertEqual((len(self.spawner.of("apps")), len(self.spawner.of("detail"))), (1, 1))

    # ---- (g) stop
    def test_stop_terminates_kills_after_the_grace_and_reaps_every_child(self):
        f = self._feed()
        self._step(f)
        self.spawner.of("index")[0].exit(0)
        self._step(f)
        stubborn = self.spawner.of("detail")[0]
        stubborn.ignores_term = True
        f.stop = True
        self.assertIs(f.stop, True)
        for p in self.spawner.procs:
            self.assertIsNotNone(p.returncode, p.args)
            self.assertTrue(p.waited, p.args)
        self.assertEqual([p.signals for p in (self.spawner.of("core")[0], self.spawner.of("apps")[0])],
                         [["TERM"], ["TERM"]])
        self.assertEqual(stubborn.signals, ["TERM", "KILL"])
        self.assertEqual(self.spawner.of("index")[0].signals, [])   # exited by itself: only reaped

    def test_an_exception_in_a_reader_still_reaps_every_child(self):
        f = self._feed()
        self._step(f)

        def boom(doc, now):
            raise RuntimeError("reader broke")
        f._detail.handler = boom
        self.spawner.of("detail")[0].write(NVQ_XML)
        with self.assertRaises(RuntimeError):
            f.run()                       # the thread body, run here: not swallowed, and cleaned up
        for p in self.spawner.procs:
            self.assertIsNotNone(p.returncode, p.args)
            self.assertTrue(p.waited, p.args)

    def test_nothing_is_launched_after_stop(self):
        f = self._feed()
        self._step(f)
        f.stop = True
        self._step(f, 1.0, 120)
        self.assertEqual(len(self.spawner.procs), 4)
        f.stop = True                                           # idempotent
        self.assertEqual(len(self.spawner.procs), 4)

    # ---- (j) tool missing
    def test_tool_missing_starts_nothing(self):
        f = self._feed(tool_ok=False)
        f.start()
        f.join(2.0)
        self.assertFalse(f.is_alive())
        f.stop = True
        self.assertEqual(self.spawner.attempts, [])


_FAKE_NVSMI = '''#!%(python)s
# a stand-in for nvidia-smi: the feed's four invocations answered from fixtures; the -l / -lms
# loops repeat every 0.2 s whatever interval was asked, until a signal ends the process
import sys, time
a = sys.argv[1:]
def out():
    if "-x" in a:
        with open(%(xml)r) as f:
            sys.stdout.write(f.read())
    elif "--query-gpu=index,pci.bus_id" in a:
        sys.stdout.write(%(index)r)
    elif any(x.startswith("--query-gpu=") for x in a):
        sys.stdout.write(%(core)r)
    elif any(x.startswith("--query-compute-apps") for x in a):
        sys.stdout.write(%(apps)r)
    sys.stdout.flush()
out()
while "-l" in a or "-lms" in a:
    time.sleep(0.2)
    out()
'''


class NvidiaFeedChildren(_Base):
    """The feed on real child processes: a fake nvidia-smi alone on PATH, everything else refused
    before it could start (R2 / §4: never the real /usr/bin/nvidia-smi)."""
    def setUp(self):
        super().setUp()
        self.bin = os.path.join(self.tmp.name, "bin")
        os.makedirs(self.bin)
        xml = os.path.join(self.tmp.name, "q.xml")
        with open(xml, "w") as f:
            f.write(NVQ_XML)
        exe = os.path.join(self.bin, "nvidia-smi")
        with open(exe, "w") as f:
            f.write(_FAKE_NVSMI % {"python": sys.executable, "xml": xml, "index": _NV_INDEX_ROWS,
                                   "core": _NV_CORE_ROWS, "apps": _NV_APPS_ROWS})
        os.chmod(exe, 0o755)
        saved_path = os.environ.get("PATH")
        os.environ["PATH"] = self.bin

        def restore_path():
            if saved_path is None:
                os.environ.pop("PATH", None)
            else:
                os.environ["PATH"] = saved_path
        self.addCleanup(restore_path)
        self.children = []
        real_popen, real_run = subprocess.Popen, subprocess.run

        def popen(args, *a, **k):
            self._check(args, k.get("executable"))
            p = real_popen(args, *a, **k)
            self.children.append(p)
            return p

        def run(args, *a, **k):
            self._check(args, k.get("executable"))
            return real_run(args, *a, **k)
        subprocess.Popen, subprocess.run = popen, run
        self.addCleanup(setattr, subprocess, "Popen", real_popen)
        self.addCleanup(setattr, subprocess, "run", real_run)

        def reap_leftovers():
            for p in self.children:
                if p.poll() is None:
                    p.kill()
                    p.wait()
        self.addCleanup(reap_leftovers)
        saved = dict(L._NV_PROC_MEM)
        L._NV_PROC_MEM.clear()
        self.addCleanup(lambda: (L._NV_PROC_MEM.clear(), L._NV_PROC_MEM.update(saved)))

    def _check(self, args, executable=None):
        """Refuse any program that is not the fake in this test's own bin directory."""
        name = executable or (args[0] if isinstance(args, (list, tuple)) else args)
        found = name if os.path.isabs(name) else shutil.which(name)
        if not (found and os.path.dirname(os.path.realpath(found)) == os.path.realpath(self.bin)):
            raise _Touched(f"program outside the test's bin: {name!r} -> {found!r}")

    def _wait_for(self, ready, limit=2.5):
        t0 = time.monotonic()
        while time.monotonic() - t0 < limit and not ready():
            time.sleep(0.05)
        return time.monotonic() - t0

    # ---- (h) --once / probe timing: the tables are there within the existing 2.5 s wait
    def test_children_deliver_the_tables_within_the_once_wait(self):
        f = L._NvidiaFeed()
        self.assertTrue(f.tool_ok)
        f.start()
        self.addCleanup(setattr, f, "stop", True)
        waited = self._wait_for(lambda: f.detail_ts and f.by_index and f.index_by_bus and L._NV_PROC_MEM)
        self.assertLess(waited, 2.5)
        self.assertEqual(f.index_by_bus, {"0000:12:00.0": 0, "0000:14:00.0": 1})
        self.assertEqual(sorted(f.by_index), [0, 1])
        self.assertEqual(f.by_index[0]["name"], "Tesla P100-PCIE-16GB")
        self.assertEqual(f.detail_by_bus["0000:14:00.0"]["pstate"], "P8")
        self.assertEqual(L._NV_PROC_MEM["3802"][0], 122)
        # the four children are V3's: three loops and one answer, nothing else
        self.assertEqual(sorted(_nv_kind(p.args) for p in self.children), ["apps", "core", "detail", "index"])
        for p in self.children:
            if _nv_kind(p.args) != "index":
                self.assertTrue("-l" in p.args or "-lms" in p.args, p.args)

    # ---- (g) on real processes: after stop, no child exists
    def test_stop_leaves_no_child_alive(self):
        f = L._NvidiaFeed()
        f.start()
        self._wait_for(lambda: len(self.children) >= 4)
        loops = [p for p in self.children if _nv_kind(p.args) != "index"]
        self.assertEqual(len(loops), 3)
        self.assertTrue(all(p.poll() is None for p in loops), "the loops are alive before stop")
        f.stop = True
        for p in self.children:
            self.assertIsNotNone(p.returncode, p.args)          # reaped: waitpid done
            with self.assertRaises(ProcessLookupError):
                os.kill(p.pid, 0)
        f.join(3.0)
        self.assertFalse(f.is_alive())

    # ---- the tripwire itself, both directions, without starting anything
    def test_tripwire_refuses_the_real_nvidia_smi_and_accepts_the_fake(self):
        with self.assertRaises(_Touched):
            self._check(["/usr/bin/nvidia-smi", "-L"])
        with self.assertRaises(_Touched):
            self._check(["nvidia-smi"], executable="/usr/bin/nvidia-smi")
        self._check(["nvidia-smi", "-q", "-x", "-l", "3"])
        self.assertEqual(os.path.realpath(shutil.which("nvidia-smi")),
                         os.path.realpath(os.path.join(self.bin, "nvidia-smi")))


if __name__ == "__main__":
    unittest.main()
