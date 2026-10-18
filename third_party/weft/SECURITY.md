# Security

Weft compiles and loads native code. Treat Python DSL files, canonical/physical
IR, compiler options, toolchain executables and saved artifacts as trusted
program inputs. The compiler and runtime are not a sandbox for untrusted code.
Loading a native artifact can execute code with the application's privileges.

The runtime validates declared buffer, ABI, target and lifetime contracts. These
checks do not replace isolation of untrusted workloads. Applications retain
responsibility for synchronization of borrowed mutable storage and for keeping
raw native-entry arguments valid throughout a call.

## Report a vulnerability

Use the repository's **Security → Report a vulnerability** action when available.
If private reporting is not enabled, open an issue asking the maintainer to
establish a private channel, without publishing exploit details, sensitive inputs
or credentials. No dedicated security mailbox or response-time guarantee is
currently advertised.

Include the affected revision, platform, toolchain, minimal reproduction and
observed impact. State whether the problem requires an untrusted artifact or can
be triggered through a valid public API call. Ordinary numerical/compiler errors
can use the normal bug report form.

Fixes are developed on the main branch. There is no separately maintained LTS
release line or guaranteed backport policy.

中文说明：Weft 会执行本机代码，不是运行不可信 DSL 或产物的沙箱。安全问题优先使用 GitHub
私密报告入口；若该入口未启用，只在公开 issue 请求建立私密沟通渠道，不公开利用细节。
