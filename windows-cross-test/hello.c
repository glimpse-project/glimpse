#include <stdio.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <mmintrin.h>

int __declspec(dllexport)
hello_func(void);

#ifdef IS_DLL
int __declspec(dllexport)
hello_func(void)
{
    return 1;
}
#else
int
main(int argc, char **argv)
{
        __m128d a, b;
        double vals[2] = {1.0, 2.0};
        a = _mm_loadu_pd (vals);
        b = _mm_add_pd (a,a);
        _mm_storeu_pd (vals,b);

	printf("Hello, World! %d, vals[0]=%f,[1]=%f\n", hello_func(), vals[0], vals[1]);
	return 0;
}
#endif

