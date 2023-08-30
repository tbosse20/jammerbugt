import shutil, os

count = 0
stack = ['../Scannings']

while len(stack) > 0:
    path = stack.pop()
    print()
    print(f'\r{path}', end=" ")
    if 'output' in path:
        print('DELETING', end=" ")
        shutil.rmtree(path)
        print('complete', end=" ")
        count += 1
        continue
    try:
        subs = os.listdir(path)
        for sub in subs:
            sub = os.path.join(path, sub)
            stack.append(sub)
    except:
        pass

print()
print(f'Finished deleting {count} "output" folder!')