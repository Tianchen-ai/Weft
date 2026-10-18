"""Function boundaries for noinline calls and nonlocal structured returns."""
import ast
import copy
import json

from triton.compiler.code_generator import flatten_values_to_ir, unflatten_ir_values


def requires_function_region(statements):
    return any(isinstance(node, (ast.For, ast.While, ast.With)) and
               any(isinstance(inner, ast.Return) for inner in ast.walk(node))
               for statement in statements for node in ast.walk(statement))


def falls_through(statements):
    for statement in statements:
        if isinstance(statement, ast.Return):
            return False
        if isinstance(statement, ast.If) and statement.orelse and not (
                falls_through(statement.body) or falls_through(statement.orelse)):
            return False
    return True


class FunctionReturns(ast.NodeTransformer):
    def visit_Return(self, node):
        result = ast.copy_location(ast.Expr(value=node.value), node)
        result._weft_helper_return = True
        return result


def compile_function_region(semantic, fn, statements):
    generator, builder = semantic.extension.generator, semantic.builder
    insertion = generator._get_insertion_point_and_loc()
    region = builder.region(())
    returns = []
    semantic.helper_returns.append(returns)
    builder.set_insertion_point_to_end(region)
    try:
        body = [FunctionReturns().visit(copy.deepcopy(statement)) for statement in statements]
        if falls_through(statements):
            body.append(FunctionReturns().visit(ast.copy_location(ast.Return(value=None), statements[-1])))
        generator.visit_compound_statement(body)
        owners = {}
        def visit(block):
            for operation in block.operations:
                if operation.name == "weft_kernel.helper_return":
                    owners[id(operation)] = block
                elif operation.name != "weft_kernel.helper":
                    for nested in operation.regions:
                        visit(nested)
        visit(region)
        # Shared SSA construction probes loop bodies and moves branch blocks.
        # Only returns in the completed construction tree belong to this call.
        returns = [(value, owners[id(operation)], operation) for value, _, operation in returns
                   if id(operation) in owners]
        result_type = generator.decide_return_type([value for value, _, _ in returns])
        for value, block, operation in returns:
            builder.block, builder.position = block, block.operations.index(operation)
            builder.set_loc(builder.create_loc(operation.location.filename, operation.location.line,
                                               operation.location.column + 1))
            value = generator.cast_to(value, result_type)
            operation.operands = tuple(flatten_values_to_ir([value]))
        builder.close_regions(region)
        types = []
        result_type._flatten_ir_types(builder, types)
    finally:
        semantic.helper_returns.pop()
        generator._set_insertion_point_and_loc(*insertion)
    result = builder.op("weft_kernel.helper", (), types,
                        {"name": json.dumps(fn.__name__), "noinline": "true" if fn.noinline else "false"}, (region,))
    handles = [result] if len(types) == 1 else list(result)
    return next(unflatten_ir_values(handles, [result_type]))
