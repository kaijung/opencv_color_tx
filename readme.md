# Color transform using OpenCV

Apply color transform based on background image's color distribution.

## using opencv-3.1.0

```
export DYLD_LIBRARY_PATH="/Users/liutao/opencv-3.1.0/build/lib:$DYLD_LIBRARY_PATH"
export DYLD_FALLBACK_LIBRARY_PATH="/Users/liutao/opencv-3.1.0/build/lib:$DYLD_FALLBACK_LIBRARY_PATH"
```

## build
```
g++ -ggdb `pkg-config --cflags --libs opencv3` color_transfer.cpp -o color_tx
```

