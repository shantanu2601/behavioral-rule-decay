"""CAPE full-report -> typed, class-tagged predicates.

One report becomes many `Predicate` rows. Every predicate carries TWO tags,
assigned AT EXTRACTION (skipping the tag would mean reprocessing later):

  field : mutex | registry | file | network | api_seq | api_set | process | command
  klass : artifact  (attacker-chosen value: mutex name, dropped filename, C2 ip)
          mechanism (required technique step: persistence path, injection k-gram)

`klass` is the INDEPENDENT VARIABLE of the whole study, so the rules that assign
it live here as explicit, auditable constants -- not learned, not hidden. Design
notes: APIs are logged at the native Nt* layer; injection is cross-process;
persistence is often an AppData drop + self-relaunch rather than a Run key;
`behavior.summary` is lossy, so facts come from the ordered calls; most call
arguments are per-run volatile, so only stable argument fields are kept.

This module only ENUMERATES candidate predicates per sample. It does not decide
which matter -- that is the miner's job downstream.
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from typing import Iterator

# ---------------------------------------------------------------------------
# Predicate record
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Predicate:
    field: str      # mutex | registry | file | network | api_seq | api_set | process | command
    klass: str      # artifact | mechanism
    ptype: str      # finer subtype, e.g. reg_persist, dropped_name, api_3gram
    value: str      # normalized identifying content

    def as_row(self, sha: str) -> dict:
        return {"sha": sha, "field": self.field, "klass": self.klass,
                "ptype": self.ptype, "value": self.value}


# ---------------------------------------------------------------------------
# Tagging knowledge (auditable constants — the study's independent variable)
# ---------------------------------------------------------------------------
# Registry paths that a persistence TECHNIQUE structurally requires (mechanism).
PERSIST_REG = re.compile(
    r"(?i)("
    r"\\currentversion\\run(once)?\b"
    r"|\\currentversion\\runservices"
    r"|\\winlogon\b"
    r"|\\explorer\\shell folders"
    r"|\\image file execution options\b"
    r"|\\appinit_dlls"
    r"|\\policies\\explorer\\run"
    r"|\\services\\[^\\]+\\imagepath"
    r"|\\currentcontrolset\\services\\"
    r")"
)

# Filesystem locations that persistence structurally requires (mechanism).
# Temp is NOT persistence (staging) -> stays artifact.
PERSIST_PATH = re.compile(
    r"(?i)("
    r"\\start menu\\programs\\startup\\"
    r"|%appdata%\\microsoft\\"
    r"|%appdata%\\[^\\]+\\"          # roaming\<dir>\ — common relaunch home
    r"|%localappdata%\\[^\\]+\\"
    r")"
)

# Command-line shapes that indicate a technique (mechanism), regardless of the
# specific paths inside them. Includes defense-evasion / impact: the technique
# imposes these exact tokens (you cannot disable Defender without naming it).
CMD_MECHANISM = re.compile(
    r"(?i)("
    r"\bcmd(\.exe)?\b.*\s/c\b"
    r"|\breg\b.*\badd\b.*\brun\b"
    r"|\bschtasks\b"
    r"|\bpowershell\b.*(-enc|-e |-w hidden|hidden|bypass)"
    r"|\brundll32\b|\bregsvr32\b|\bmshta\b"
    r"|\bwscript\b|\bcscript\b"
    # --- defense evasion / impact (technique-imposed) ---
    r"|\bsc\b.*\b(stop|delete|config)\b"
    r"|\bnet\b\s+stop\b|\btaskkill\b"
    r"|\bset-mppreference\b|\badd-mppreference\b|\bmpcmdrun\b"
    r"|windows\s+defender|windefend"
    r"|\bnetsh\b.*\b(firewall|advfirewall)\b"
    r"|\bvssadmin\b.*\bdelete\b|\bbcdedit\b|\bwbadmin\b.*\bdelete\b"
    r"|\bwmic\b.*\bshadowcopy\b.*\bdelete\b"
    r"|\bwevtutil\b.*\b(cl|clear)\b|\bclear-eventlog\b"
    r"|\battrib\b.*\+[hs]|\bcipher\b\s+/w"
    r")"
)

# Registry writes that a defense-evasion technique imposes (AV/Defender tamper,
# security-tool disabling). Mechanism, not attacker-chosen artifact.
DEFENSE_TAMPER_REG = re.compile(
    r"(?i)("
    r"\\windows defender\\|\\microsoft\\windows defender"
    r"|\\real-time protection\\"
    r"|disableantispyware|disableantivirus|disablerealtimemonitoring"
    r"|disablebehaviormonitoring|disableonaccessprotection|disablescanonrealtimeenable"
    r"|disablenotifications|tamperprotection"
    r"|disabletaskmgr|disableregistrytools"
    r")"
)


def classify_command(nc: str) -> tuple[str, str]:
    """(ptype, klass) for a normalized command string. Single source of truth."""
    if CMD_MECHANISM.search(nc):
        return "cmd_technique", "mechanism"
    return "cmd", "artifact"


def classify_registry(nk: str) -> tuple[str, str]:
    """(ptype, klass) for a normalized registry key. Benign keys are dropped
    upstream, so this only sees persistence / tamper / plain writes."""
    if PERSIST_REG.search(nk):
        return "reg_persist", "mechanism"
    if DEFENSE_TAMPER_REG.search(nk):
        return "reg_defense_tamper", "mechanism"
    return "reg_write", "artifact"

# APIs whose ordered occurrence is a technique step. Grouped so variants of the
# SAME concept can optionally be canonicalized (APIs are logged at the Nt* layer;
# we do NOT hardcode a detection template; this only informs tagging/canonicalization).
INJECTION_APIS = {
    "NtAllocateVirtualMemory", "VirtualAllocEx", "VirtualAlloc",
    "NtWriteVirtualMemory", "WriteProcessMemory",
    "NtMapViewOfSection", "NtUnmapViewOfSection",
    "NtCreateThreadEx", "CreateRemoteThread", "RtlCreateUserThread",
    "NtOpenProcess", "OpenProcess", "SetThreadContext", "QueueUserAPC",
    "NtResumeThread", "ResumeThread",
}

# Argument fields that are STABLE/semantic and safe to key a predicate on.
# Everything else (handles, addresses, pids, thread ids, timestamps) is volatile
# per-run and must be dropped (lesson #5).
STABLE_ARG_FIELDS = {
    "regkey", "FullName", "KeyHandle",          # registry key strings vary; keep the path ones
    "CommandLine", "ApplicationName", "FilePath", "Filepath", "filepath",
    "MutexName", "name",
}

# APIs that create a process (for the command / spawn predicates).
PROC_CREATE_APIS = {"NtCreateUserProcess", "CreateProcessInternalW",
                    "CreateProcessA", "CreateProcessW", "ShellExecuteExW", "WinExec"}
# APIs that write the registry (extract the key argument).
REG_WRITE_APIS = {"RegSetValueExA", "RegSetValueExW", "NtSetValueKey",
                  "RegCreateKeyExA", "RegCreateKeyExW"}

# noise mutexes present in nearly every sandbox run (OS/runtime, not attacker-chosen)
BENIGN_MUTEX = {"RasPbFile", "Local\\!IETld!Mutex"}
BENIGN_MUTEX_SUB = re.compile(
    r"(?i)("
    r"MSCTF|CicLoad|IESQMMUTEX|SQMMUTEX|WininetStartup"
    r"|Zones?\w*CounterMutex|ZoneAttributeCache"   # IE security-zone cache objects
    r"|\.net clr|CLR_CASOFF"                        # .NET runtime mutexes
    r"|DefaultTabtip|LanmanDatagramReceiver"        # OS UI / SMB browser objects
    r")"
)

# Registry trees that any Windows process touches — OS chatter, NOT attacker
# artifacts. Dropped entirely (persistence keys never live here, so PERSIST_REG
# is unaffected). Auditable, like the tagging constants.
BENIGN_REG = re.compile(
    r"(?i)("
    r"\\explorer\\fileexts\\"
    r"|\\internet settings\\"
    r"|\\microsoft\\tracing\\"
    r"|\\action center\\"
    r"|\\wpad\\|\\wpadlast"
    r"|\\explorer\\shell folders\\?$"       # the plain shell-folders enum (not persistence)
    r"|\\mucache\\|\\muicache\\"
    r"|\\shellnoroam\\|\\bagmru\\|\\bags\\"
    r"|\\vb and vba program settings\\"
    r"|\\napagent\\"                            # Network Access Protection (OS)
    r"|\\policies\\microsoft\\windows\\ipsec\\" # IPSEC policy (OS)
    r"|\\fusion\\|\\gacchangenotification"      # .NET Fusion / GAC (runtime)
    r"|see_mask_nozonecheck"                    # zone-check suppression (env, ubiquitous)
    r")"
)

# obvious non-C2 infrastructure IPs (DNS / update endpoints); minimal by design
BENIGN_IP = {"8.8.8.8", "8.8.4.4", "1.1.1.1", "127.0.0.1", "0.0.0.0"}

# --- OS/sandbox background that leaks past the per-field filters above. These
# are present because of the Windows/2021-sandbox ENVIRONMENT, not attacker
# choice, and surface as spuriously "durable" artifacts (present in every run).
# Auditable, like the tagging constants; applied at parse time and by refilter.py.
BENIGN_FILE = re.compile(
    r"(?i)("
    r"cryptneturlcache"                         # Windows certificate URL cache
    r"|\\inetcache\\|temporary internet files"  # IE/Edge web cache
    r"|(^|\\)index\.dat$"                        # IE cache index
    r"|setupapi\.(app|dev)\.log$"               # OS device-install logs
    r"|^\\device\\"                              # raw device objects, not dropped files
    r"|lanmandatagramreceiver"                  # SMB browser object
    r"|@[^\\/]+\[\d+\]\.(txt|htm|html)$"        # IE web cache: user@host[n].txt
    r"|^[0-9a-f]{32}$"                           # cert/INetCache content (hash-named)
    r")"
)
# benign network infrastructure live in the 2021 sandbox (CDN, cert revocation,
# connectivity checks, telemetry) — survives for benign reasons, faking
# durability. Dead 2017-2019 C2 does NOT survive, so filtering here is low-risk
# and directionally correct; noted as a stated, auditable choice in the paper.
BENIGN_NET_DOMAIN = re.compile(
    r"(?i)("
    r"lencr\.org|ocsp\.|crl\.|digicert|verisign|symc[bd]"          # cert / OCSP / CRL
    r"|ip-adress\.com|whatismyip|checkip|ipinfo\.io|myexternalip"  # ip-check services
    r"|msftncsi|msftconnecttest|windowsupdate"                     # MS connectivity
    r"|\.microsoft\.com$|\.windows\.com$|\.msn\.com$"              # Microsoft
    r"|(^|\.)google\.com$|googleapis|gstatic|gvt1\.com|\.bing\.com$"  # Google / Bing
    r")"
)
# benign IP prefixes: Google + Akamai edge (104.64.0.0/10) + Microsoft/Bing.
_BENIGN_IP_PREFIX = (
    "172.217.", "173.194.", "216.58.", "142.250.", "74.125.",   # Google
    "13.107.", "204.79.197.",                                    # Microsoft / Bing
)
def is_benign_ip(ip: str) -> bool:
    if ip in BENIGN_IP or ip.startswith(_BENIGN_IP_PREFIX):
        return True
    try:
        a, b, *_ = (int(x) for x in ip.split("."))
    except ValueError:
        return False
    return a == 23 or (a == 104 and 64 <= b <= 127)   # Akamai edge blocks

# path fragments that mark a named pipe, not a dropped file
PIPE_RX = re.compile(r"(?i)(\\\?\?\\pipe\\|\\\\\.\\pipe\\|\\pipe\\)")

K_GRAM = 3  # ordered api-sequence length


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
_ENV = [
    (re.compile(r"(?i)c:\\users\\[^\\]+\\appdata\\local\\temp"), "%TEMP%"),
    (re.compile(r"(?i)c:\\users\\[^\\]+\\appdata\\local"), "%LOCALAPPDATA%"),
    (re.compile(r"(?i)c:\\users\\[^\\]+\\appdata\\roaming"), "%APPDATA%"),
    (re.compile(r"(?i)c:\\users\\[^\\]+"), "%USERPROFILE%"),
    (re.compile(r"(?i)c:\\windows\\system32"), "%SYSTEM32%"),
    (re.compile(r"(?i)c:\\windows"), "%WINDIR%"),
    (re.compile(r"(?i)c:\\programdata"), "%PROGRAMDATA%"),
]


def norm_path(p: str) -> str:
    if not p:
        return p
    p = p.strip().strip('"')
    for rx, tok in _ENV:
        p = rx.sub(tok, p)
    return p


# Auto-run roots a persistence drop can land in. A drop-persist MECHANISM keys on
# WHICH auto-run location the family drops into (stable across campaigns), NOT on
# the specific sub-folder it picks (rotates per campaign -> that is an artifact).
# This fixes the granularity bug where `%LOCALAPPDATA%\iproppass\` was tagged a
# mechanism yet decayed like the artifact `iproppass` inside it.
_PERSIST_ROOTS = ("%APPDATA%", "%LOCALAPPDATA%", "%PROGRAMDATA%", "%USERPROFILE%")


def persist_location_class(np: str) -> str:
    """Collapse a persistence drop path to its auto-run LOCATION CLASS."""
    low = np.lower()
    if "\\start menu\\programs\\startup\\" in low:
        return "%STARTUP%\\"
    for root in _PERSIST_ROOTS:
        if low.startswith(root.lower() + "\\"):
            return f"{root}\\*\\"   # keep the root; the sub-folder is an artifact
    return np                       # other PERSIST_PATH hits are already generic


# --- Mechanism-value normalization (uniform granularity rule) -----------------
# A MECHANISM predicate must key on the technique skeleton, not on the volatile
# campaign tokens embedded in the raw value (random Run value-names, scheduled-
# task GUIDs, drop paths/filenames, self-relaunch exe names). Leaving them in
# makes a real mechanism decay like the artifact inside it (see §5.4 / the
# %LOCALAPPDATA%\iproppass\ and schtasks {GUID} cases). Applied ONLY to
# mechanism-class predicates; artifacts keep their exact value by definition.
_REG_RUN = re.compile(r"(?i)(\\currentversion\\run[a-z]*)\\.+$")
_REG_IFEO = re.compile(r"(?i)(\\image file execution options)\\[^\\]+")
_GUID = re.compile(r"\{?[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\}?")
_USERPATH = re.compile(
    r"(?i)%(APPDATA|LOCALAPPDATA|TEMP|PROGRAMDATA|USERPROFILE)%(?:\\[^\"\s]*)?")
_LONGHEX = re.compile(r"\b[0-9a-fA-F]{16,}\b")


def normalize_registry_mech(nk: str) -> str:
    """Collapse the campaign-random value-name off a persistence key."""
    nk = _REG_RUN.sub(r"\1\\*", nk)
    nk = _REG_IFEO.sub(r"\1\\*", nk)
    return nk


def normalize_command_mech(nc: str) -> str:
    """Strip volatile tokens (GUIDs, user-writable drop paths, long hashes),
    keeping the technique skeleton (LOLBin under %SYSTEM32%/%WINDIR%, flags)."""
    nc = _GUID.sub("{*}", nc)
    nc = _USERPATH.sub(lambda m: f"%{m.group(1).upper()}%\\*", nc)
    nc = _LONGHEX.sub("*", nc)
    return nc


def canon_mech_value(field: str, ptype: str, value: str) -> str:
    """Single source of truth: canonical value for a mechanism predicate."""
    if field == "registry":
        return normalize_registry_mech(value)
    if field == "command":
        return normalize_command_mech(value)
    if field == "process":
        if ptype == "self_relaunch":
            return "<self>"                       # the exe name is an artifact
        if ptype == "spawn_lolbin" and ">" in value:
            return "*>" + value.split(">", 1)[1]  # keep the child LOLBin only
    if field == "file" and ptype == "drop_persist_dir":
        return persist_location_class(value)
    return value


def _args_dict(call: dict) -> dict:
    a = call.get("arguments")
    if isinstance(a, list):
        return {x.get("name"): x.get("value") for x in a if isinstance(x, dict)}
    if isinstance(a, dict):
        return a
    return {}


# ---------------------------------------------------------------------------
# Per-field extractors
# ---------------------------------------------------------------------------
def _mutexes(summary: dict) -> Iterator[Predicate]:
    for m in summary.get("mutexes") or []:
        if not m or m in BENIGN_MUTEX or BENIGN_MUTEX_SUB.search(m):
            continue
        yield Predicate("mutex", "artifact", "mutex", m)


def _registry(summary: dict, reg_write_keys: set[str]) -> Iterator[Predicate]:
    # summary keys (lossy) + keys observed being written via calls
    keys = set(summary.get("write_keys") or []) | reg_write_keys
    for k in keys:
        if not k:
            continue
        nk = norm_path(k)
        if BENIGN_REG.search(nk) and not PERSIST_REG.search(nk):
            continue  # OS chatter, not an attacker artifact
        ptype, klass = classify_registry(nk)
        val = normalize_registry_mech(nk) if klass == "mechanism" else nk
        yield Predicate("registry", klass, ptype, val)


def _files(summary: dict, dropped: list) -> Iterator[Predicate]:
    paths: list[str] = []
    for d in dropped or []:
        for gp in d.get("guest_paths") or []:
            paths.append(gp)
    paths += summary.get("write_files") or []
    for raw in paths:
        if not raw or PIPE_RX.search(raw):
            continue
        np = norm_path(raw)
        if BENIGN_FILE.search(np) or BENIGN_FILE.search(np.rsplit("\\", 1)[-1]):
            continue  # OS/sandbox background (cert cache, web cache, device objects)
        # directory (location) vs filename
        parts = np.rsplit("\\", 1)
        dirp = parts[0] + "\\" if len(parts) == 2 else np
        name = parts[1] if len(parts) == 2 else ""
        if PERSIST_PATH.search(np):
            loc = persist_location_class(np)
            # mechanism = the auto-run LOCATION CLASS (stable across campaigns)
            yield Predicate("file", "mechanism", "drop_persist_loc", loc)
            # the specific sub-folder the family picked is attacker-chosen
            if dirp and dirp != loc:
                yield Predicate("file", "artifact", "drop_dir", dirp)
        else:
            yield Predicate("file", "artifact", "drop_dir", dirp)
        if name:
            yield Predicate("file", "artifact", "drop_name", name)


def _network(net: dict) -> Iterator[Predicate]:
    for h in net.get("hosts") or []:
        ip = h.get("ip") if isinstance(h, dict) else h
        if ip and not is_benign_ip(str(ip)):
            yield Predicate("network", "artifact", "host_ip", str(ip))
    for dom in net.get("domains") or []:
        d = dom.get("domain") if isinstance(dom, dict) else dom
        if d and not BENIGN_NET_DOMAIN.search(str(d)):
            yield Predicate("network", "artifact", "domain", str(d))


def _commands(summary: dict) -> Iterator[Predicate]:
    for c in summary.get("executed_commands") or []:
        if not c:
            continue
        nc = norm_path(c)
        ptype, klass = classify_command(nc)
        val = normalize_command_mech(nc) if klass == "mechanism" else nc
        yield Predicate("command", klass, ptype, val)


def _process_structure(processes: list) -> Iterator[Predicate]:
    by_pid = {p.get("process_id"): p for p in processes}
    for p in processes:
        parent = by_pid.get(p.get("parent_id"))
        if parent is None:
            continue
        cn = (p.get("process_name") or "").lower()
        pn = (parent.get("process_name") or "").lower()
        if cn and cn == pn:
            # self-relaunch is the mechanism; the exe name is an artifact
            yield Predicate("process", "mechanism", "self_relaunch", "<self>")
        if cn in {"cmd.exe", "powershell.exe", "wscript.exe", "cscript.exe",
                  "rundll32.exe", "regsvr32.exe", "mshta.exe", "schtasks.exe"}:
            # keep the child LOLBin; the parent name is an artifact
            yield Predicate("process", "mechanism", "spawn_lolbin", f"*>{cn}")


def _api_from_calls(processes: list) -> Iterator[Predicate]:
    reg_keys: set[str] = set()
    for p in processes:
        calls = p.get("calls") or []
        # collapse consecutive identical apis to kill stall-loop noise
        seq: list[str] = []
        for c in calls:
            api = c.get("api")
            if not api:
                continue
            if not seq or seq[-1] != api:
                seq.append(api)
            # side-harvest: registry-write keys and proc-create command lines
            if api in REG_WRITE_APIS:
                a = _args_dict(c)
                k = a.get("regkey") or a.get("FullName")
                if k:
                    reg_keys.add(k)
        # ordered k-grams (dedup within sample)
        seen_grams: set[tuple] = set()
        for i in range(len(seq) - K_GRAM + 1):
            g = tuple(seq[i:i + K_GRAM])
            if g in seen_grams:
                continue
            seen_grams.add(g)
            yield Predicate("api_seq", "mechanism", f"api_{K_GRAM}gram", ">".join(g))
    # expose harvested reg keys via a sentinel predicate stream consumer
    _api_from_calls.reg_keys = reg_keys  # type: ignore[attr-defined]


def _api_set(summary: dict) -> Iterator[Predicate]:
    for api in summary.get("resolved_apis") or []:
        if api in INJECTION_APIS:
            yield Predicate("api_set", "mechanism", "resolved_api", api)
        # non-injection resolved apis omitted in v1 (too noisy); revisit


# ---------------------------------------------------------------------------
# Top-level: report -> predicates
# ---------------------------------------------------------------------------
def extract_predicates(report: dict) -> list[Predicate]:
    b = report.get("behavior") or {}
    summary = b.get("summary") or {}
    processes = b.get("processes") or []
    dropped = report.get("dropped") or []
    net = report.get("network") or {}

    preds: list[Predicate] = []
    # api extractor also harvests registry-write keys as a side effect
    preds += list(_api_from_calls(processes))
    reg_write_keys = getattr(_api_from_calls, "reg_keys", set())

    preds += list(_mutexes(summary))
    preds += list(_registry(summary, reg_write_keys))
    preds += list(_files(summary, dropped))
    preds += list(_network(net))
    preds += list(_commands(summary))
    preds += list(_process_structure(processes))
    preds += list(_api_set(summary))

    # dedup identical predicates within a sample
    return list(dict.fromkeys(preds))


# ---------------------------------------------------------------------------
# Streaming report reader (random access from the 17 GB zip)
# ---------------------------------------------------------------------------
def iter_reports(zip_path, limit=None, shas=None):
    """Yield (sha, report_dict). If `shas` given, read just those; else stream."""
    import json
    z = zipfile.ZipFile(zip_path)
    names = [n for n in z.namelist() if n.endswith(".json")]
    if shas is not None:
        want = set(shas)
        names = [n for n in names if n.split("/")[-1][:-5] in want]
    if limit is not None:
        names = names[:limit]
    for n in names:
        sha = n.split("/")[-1][:-5]
        try:
            yield sha, json.loads(z.read(n))
        except Exception as e:  # noqa: BLE001
            print(f"  !! failed {sha[:16]}: {e}")
