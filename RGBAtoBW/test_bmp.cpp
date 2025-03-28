#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <assert.h>
#include "test_bmp.h"
#include <memory>
#include <cstring>
#include <iostream>
#include <array>

/**
 *  pixel = data[col + row * width];
 *  unsigned char a = (pixel >> 24) & 0xff;
 *  unsigned char r = (pixel >> 16) & 0xff;
 *  unsigned char g = (pixel >> 8) & 0xff;
 *  unsigned char b = pixel & 0xff;
 *
 *  if uint32_t pixel = 0xAABBCCDD;
 *  Alpha is 0xAA。
 *  R is 0xBB
 *  G is 0xCC
 *  B is 0xDD
 *  pixel >> 24 will get 0x000000AA
 *  0x000000AA & 0xFF will get 0xAA
 *
 */

void bmpLoad(BMP *bmp, const char *fileName)
{
    FILE *file;
    BYTE header[14];
    BYTE info[40];

    if ((file = fopen(fileName, "rb")) == NULL)
    {
        fprintf(stderr, "Error: bmpLoad(), File open fail!\n");
        exit(EXIT_FAILURE);
    }

    fread(header, 1, HEADER_SIZE, file);
    fread(info, 1, INFO_SIZE, file);
    memcpy(bmp->header, header, HEADER_SIZE);
    memcpy(bmp->info, info, INFO_SIZE);

    // Header
    bmp->signature = B2U16(header, 0);
    assert(bmp->signature == 0x4D42);
    bmp->fileSize = B2U32(header, 2);
    bmp->dataOffset = B2U32(header, 10);

    // InfoHeader
    bmp->size = B2U32(info, 0);
    assert(bmp->size == 40);
    bmp->width = B2U32(info, 4);
    bmp->height = B2U32(info, 8);
    bmp->planes = B2U16(info, 12);
    assert(bmp->planes == 1);
    bmp->bitsPerPixel = B2U16(info, 14);
    assert(bmp->bitsPerPixel == 32);
    bmp->compression = B2U32(info, 16);
    bmp->imageSize = B2U32(info, 20);
    bmp->xPixelsPerM = B2U32(info, 24);
    bmp->yPixelsPerM = B2U32(info, 28);
    bmp->colorsUsed = B2U32(info, 32);
    bmp->colorsImportant = B2U32(info, 36);

    // Image data
    bmp->data = (uint32_t *)malloc(DataSize(bmp));
    fseek(file, bmp->dataOffset, SEEK_SET);
    fread(bmp->data, 1, DataSize(bmp), file);
    fclose(file);
}

void bmpPrint(BMP *bmp)
{
    printf("==== Header ====\n");
    printf("Signature = %04X\n", bmp->signature); // 0x4d42 = BM
    printf("FileSize = %u \n", bmp->fileSize);
    printf("DataOffset = %u \n", bmp->dataOffset);
    printf("==== Info ======\n");
    printf("Info size = %u \n", bmp->size);
    printf("Width = %u \n", bmp->width);
    printf("Height = %u \n", bmp->height);
    printf("BitsPerPixel = %u \n", bmp->bitsPerPixel);
    printf("Compression = %u \n", bmp->compression);
    printf("================\n");
}

void bmpSave(BMP *bmp, const char *fileName)
{
    FILE *file;

    if ((file = fopen(fileName, "wb")) == NULL)
    {
        fprintf(stderr, "Error: bmpSave(), File create fail!\n");
        exit(EXIT_FAILURE);
    }

    fwrite(bmp->header, 1, HEADER_SIZE, file);
    fwrite(bmp->info, 1, INFO_SIZE, file);
    fseek(file, bmp->dataOffset, SEEK_SET);
    fwrite(bmp->data, 1, DataSize(bmp), file);
    fclose(file);

    printf("Save the picture successfully!\n");
}

void rgbaToBw(BMP *bmp, int width, int height, long stride)
{
    int row, col;
    uint32_t pixel, r, g, b, a, bw;
    uint32_t *data = bmp->data;

    for (row = 0; row < height; row++)
    {
        for (col = 0; col < width; col++)
        {
            pixel = data[col + row * stride / 4];
            a = (pixel >> 24) & 0xff;
            r = (pixel >> 16) & 0xff;
            g = (pixel >> 8) & 0xff;
            b = pixel & 0xff;
            bw = (uint32_t)(r * 0.299 + g * 0.587 + b * 0.114);
            data[col + row * stride / 4] = (a << 24) + (bw << 16) + (bw << 8) + (bw);
        }
    }
}
