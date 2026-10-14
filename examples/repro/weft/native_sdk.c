#include <weft/runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void checked(weft_status status, const weft_error *error) {
    if (status != WEFT_OK) {
        fprintf(stderr, "%s: %s\n", weft_status_name(status), error->message);
        exit(1);
    }
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) return 2;
    weft_error error;
    weft_context *context;
    checked(weft_context_create(&context, &error), &error);
    weft_package *package;
    checked(weft_package_open(context, argv[1], &package, &error), &error);
    weft_kernel *kernel;
    checked(weft_package_kernel(package, argc == 3 ? argv[2] : "affine", &kernel, &error), &error);
    float input[129], output[129];
    for (int i = 0; i < 129; ++i) input[i] = (float)i / 17;
    size_t shape = 129;
    weft_value values[4] = {0};
    values[0].kind = WEFT_VIEW;
    values[0].view = (weft_view){input, sizeof(input), "dense.f32", 1, &shape, NULL, true};
    values[1].kind = WEFT_VIEW;
    values[1].view = (weft_view){output, sizeof(output), "dense.f32", 1, &shape, NULL, false};
    values[2].kind = values[3].kind = WEFT_SCALAR;
    values[2].dtype = values[3].dtype = WEFT_F32;
    values[2].scalar.f32 = 1.25f;
    values[3].scalar.f32 = -0.5f;
    weft_call *call;
    checked(weft_call_prepare(kernel, values, 4, NULL, 0, &call, &error), &error);
    weft_kernel_release(kernel);
    weft_package_release(package);
    weft_context_release(context);
    checked(weft_call_invoke(call, &error), &error);
    for (int i = 0; i < 129; ++i) {
        float expected = input[i] * 1.25f - 0.5f;
        if (!isfinite(output[i]) || fabsf(output[i] - expected) > 1e-4f + 2e-3f * fabsf(expected)) return 1;
    }
    /* Prepared metadata is independent of the caller's descriptor lifetime. */
    shape = 1;
    memset(values, 0, sizeof(values));
    weft_named_value update = {0};
    update.name = "scale";
    update.value.kind = WEFT_SCALAR;
    update.value.dtype = WEFT_F32;
    update.value.scalar.f32 = 2.0f;
    weft_call *rebound;
    checked(weft_call_rebind(call, &update, 1, NULL, 0, &rebound, &error), &error);
    weft_call_release(call);
    checked(weft_call_invoke(rebound, &error), &error);
    for (int i = 0; i < 129; ++i) {
        float expected = input[i] * 2.0f - 0.5f;
        if (!isfinite(output[i]) || fabsf(output[i] - expected) > 1e-4f + 2e-3f * fabsf(expected)) return 1;
    }
    weft_call *invalid = (weft_call *)1;
    update.name = "absent";
    if (weft_call_rebind(rebound, &update, 1, NULL, 0, &invalid, &error) != WEFT_INVALID_ARGUMENT || invalid != NULL)
        return 1;
    weft_call_release(rebound);
    puts("native-c: affine, independent handle lifetime and descriptor-free rebind passed");
    return 0;
}
