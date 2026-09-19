obj-m += chardev.o
KDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(CURDIR)

all: module test

module:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

test: test.c
	$(CC) -O2 test.c -o test

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f test
