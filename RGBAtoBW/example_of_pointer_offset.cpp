#include <iostream>
#include <cstdint>

typedef unsigned char BYTE;

int main()
{
    // Example RGBA pixel data
    uint32_t data[] = {
        0x11223344,
        0x55667788,
        0x99AABBCC,
        0xDDEEFF00};

    size_t idx = 2; // Index of the pixel we want to access (Pixel 2)

    // Basic pointer arithmetic to get the address of the pixel
    uint32_t *pixel_address = data + idx;
    BYTE *byte_address = (BYTE *)pixel_address;

    // Print all data array values in hexadecimal format
    std::cout << "Data array values:" << std::endl;
    for (int i = 0; i < 4; ++i)
    {
        std::cout << "Pixel " << i << ": 0x" << std::hex << data[i] << std::endl;
    }

    // Printing address of data and data+idx
    std::cout << "\nBase address of data array: " << static_cast<void *>(data) << std::endl;
    std::cout << "Address of data[" << idx << "]: " << static_cast<void *>(data + idx) << std::endl;

    // Break down the BYTE-pointers to individual components of Pixel 2 (0x99AABBCC)
    BYTE r = *(byte_address + 2); // Most significant byte
    BYTE g = *(byte_address + 1); // Next byte
    BYTE b = *(byte_address);     // Least significant byte
    BYTE a = *(byte_address + 3); // Alpha channel (last byte)

    // Printing the values
    std::cout << "\nPixel " << idx << " RGBA values:" << std::endl;
    std::cout << "R: 0x" << std::hex << static_cast<int>(r) << std::endl;
    std::cout << "G: 0x" << std::hex << static_cast<int>(g) << std::endl;
    std::cout << "B: 0x" << std::hex << static_cast<int>(b) << std::endl;
    std::cout << "A: 0x" << std::hex << static_cast<int>(a) << std::endl;

    return 0;
}
/**
 * Data array values:
 * Pixel 0: 0x11223344
 * Pixel 1: 0x55667788
 * Pixel 2: 0x99aabbcc
 * Pixel 3: 0xddeeff00
 *
 * Base address of data array: 0x7ffee7355a40
 * Address of data[2]: 0x7ffee7355a48
 * Pixel 2 RGBA values:
 * R: 0x99
 * G: 0xaa
 * B: 0xbb
 * A: 0xcc
 */