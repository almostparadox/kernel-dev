#include <linux/fs.h>
#include <linux/init.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/uaccess.h>

#define BUF_SIZE 256

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Learner");
MODULE_DESCRIPTION("Minimal misc character device");
MODULE_VERSION("0.1");

// ponytail: static single buffer; upgrade to circular buffer + mutex for multi-thread
static char dev_buffer[BUF_SIZE];
static size_t data_len;

static ssize_t dev_read(struct file *file, char __user *buf, size_t count, loff_t *ppos)
{
    if (*ppos >= data_len)
        return 0;

    if (count > data_len - *ppos)
        count = data_len - *ppos;

    if (copy_to_user(buf, dev_buffer + *ppos, count))
        return -EFAULT;

    *ppos += count;
    return count;
}

static ssize_t dev_write(struct file *file, const char __user *buf, size_t count, loff_t *ppos)
{
    if (count > BUF_SIZE - 1)
        count = BUF_SIZE - 1;

    if (copy_from_user(dev_buffer, buf, count))
        return -EFAULT;

    dev_buffer[count] = '\0';
    data_len = count;
    *ppos += count;
    return count;
}

static const struct file_operations dev_fops = {
    .owner = THIS_MODULE,
    .read = dev_read,
    .write = dev_write,
};

static struct miscdevice sample_device = {
    .minor = MISC_DYNAMIC_MINOR,
    .name = "simple_chardev",
    .fops = &dev_fops,
};

static int __init chardev_init(void)
{
    pr_info("simple_chardev: registered\n");
    return misc_register(&sample_device);
}

static void __exit chardev_exit(void)
{
    misc_deregister(&sample_device);
    pr_info("simple_chardev: unregistered\n");
}

module_init(chardev_init);
module_exit(chardev_exit);
