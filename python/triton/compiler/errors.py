import ast
from typing import Optional
from ..errors import TritonError


class CompilationError(TritonError):
    """Base class for all errors raised during compilation"""
    source_line_count_max_in_message = 12

    def _format_message(self) -> str:
        node = self.node
        if self.src is None:
            source_excerpt = " <source unavailable>"
        else:
            if hasattr(node, 'lineno'):
                source_excerpt = self.src.split('\n')[:node.lineno][-self.source_line_count_max_in_message:]
                if source_excerpt:
                    source_excerpt.append(' ' * node.col_offset + '^')
                    source_excerpt = '\n'.join(source_excerpt)
                else:
                    source_excerpt = " <source empty>"
            else:
                source_excerpt = self.src

        message = "at {}:{}:\n{}".format(node.lineno, node.col_offset, source_excerpt) if hasattr(
            node, 'lineno') else source_excerpt
        if self.error_message:
            message += '\n' + self.error_message
        return message

    def __init__(self, src: Optional[str], node: ast.AST, error_message: Optional[str] = None):
        self.src = src
        self.node = node
        self.error_message = error_message
        self.message = self._format_message()

    def __str__(self):
        return self.message

    def __reduce__(self):
        # this is necessary to make CompilationError picklable
        return type(self), (self.src, self.node, self.error_message)


class CompileTimeAssertionFailure(CompilationError):
    """Specific exception for failed tests in `static_assert` invocations"""
    pass


class UnsupportedLanguageConstruct(CompilationError):
    pass


class BackendCompilationError(TritonError):
    def __init__(self, report):
        self.report = report
        self.candidate_rejected = report["category"] in ("unsupported", "resource-limit")
        messages = "\n".join(f"{item['location']}: {item['message']}" for item in report["diagnostics"])
        super().__init__(f"{report['stage']} / {report['pass']} ({report['category']}):\n{messages}")

    def __reduce__(self):
        # Framework compilation may cross a worker-process boundary. Rebuild
        # from the structured report, not Exception's formatted string args.
        return type(self), (self.report,)
