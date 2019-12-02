

#include <Python.h>
#include <numpy/arrayobject.h>


//
static PyObject* blend(PyObject* self, PyObject* args){
    npy_intp dims[4];
    dims[0] = 10;
    dims[1] = 10;
    dims[2] = 2; // two direction x,y
    dims[3] = 1;
    PyObject *mv = PyArray_Zeros(4,dims,NPY_INT32, 0);
    printf("hello c+python!\n");
//    PyObject* mvs[2];
//    for(int i=0;i<2;i++){
//        PyObject *mv = PyArray_Zeros(4,dims,NPY_INT32, 0);
//        mvs[i] = mv;
//    }
//    PyObject *ans = PyArray_Concatenate(mvs,3);

    return mv;
}

//声明模块中的方法表
static PyMethodDef Methods[] = {
    {"blend", blend, METH_VARARGS,
         "test"},
    {NULL, NULL, 0, NULL}
};

// 模块结构
static struct PyModuleDef sjtestmodule = {
    PyModuleDef_HEAD_INIT,
    "sjtest",   /* name of module */
    NULL,       /* module documentation, may be NULL */
    -1,         /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    Methods // Methods table
};


PyMODINIT_FUNC PyInit_sjtest(void)
{
    PyObject *m;

    m = PyModule_Create(&sjtestmodule);
    if (m == NULL)
        return NULL;

    /* IMPORTANT: this must be called */
    import_array();

    return m;
}

int main(int argc, char *argv[])
{

    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    /* Add a built-in module, before Py_Initialize */
    PyImport_AppendInittab("sjtest", PyInit_sjtest);

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    PyMem_RawFree(program);
    return 0;
}
