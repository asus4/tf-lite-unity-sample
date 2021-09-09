
#include <iostream>
#include <cstdarg>
#include <cstring>

extern "C" {

const char* UnityTFLiteStringFormat(const char *format, va_list argsOriginal) {
    char buffer[512];
    va_list args;
    va_copy(args, argsOriginal);
    int length = vsprintf(buffer, format, args);
    va_end(args);

    // copy string
    char* msg = (char*)malloc(length + 1);
    strcpy(msg, buffer);
    // msg is dalloced in Unity
    return msg;
}

}
