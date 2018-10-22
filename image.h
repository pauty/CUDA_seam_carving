#ifndef IMAGE
#define IMAGE

typedef struct pixel{
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;   
}pixel;

typedef struct rgba_image{
    int w;
    int h;
    pixel *pixels;
}rgba_image;

#endif
