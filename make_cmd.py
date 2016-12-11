#!/usr/bin/env python


cache_keys = []

with open('cache_key.csv') as f:
    for line in f.readlines():
        cache_keys.append(line.strip())

num = len(cache_keys) / 20
remain = len(cache_keys) % 20

if remain > 0:
    num += 1

cmds = []
for i in xrange(num):
    cmds.append('del %s\n\n' % ' '.join(cache_keys[(20 * i):(20*(i+1))]))

output = open('cmds.txt', 'wb')
for c in cmds:
    output.write(c)

output.close()
