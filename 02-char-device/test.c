#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main(void)
{
    const char msg[] = "hello from userland!";
    char buf[64] = {0};

    int fd = open("/dev/simple_chardev", O_RDWR);
    if (fd < 0) {
        perror("open /dev/simple_chardev");
        return 1;
    }

    ssize_t written = write(fd, msg, strlen(msg));
    assert(written == (ssize_t)strlen(msg));

    ssize_t read_bytes = read(fd, buf, sizeof(buf) - 1);
    assert(read_bytes == written);
    assert(strcmp(buf, msg) == 0);

    close(fd);
    printf("test passed: wrote and read '%s'\n", buf);
    return 0;
}
