package com.example.game_test;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;

import androidx.annotation.NonNull;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.androidgamesdk.GameActivity;
import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends GameActivity {
    private static final String TAG = "CameraActivity";
    private static final int PERMISSION_REQUEST_CODE = 100;
    private ExecutorService cameraExecutor;

    static {
        System.loadLibrary("game_test");
    }

    public native void processCameraFrameDirect(ByteBuffer data, int width, int height, int rotationDegrees);

    /** 加载 OpenCV DNN 模型（绝对路径，一般为应用私有目录下的 ONNX 文件） */
    public static native boolean nativeInitDnn(String modelAbsolutePath);

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        super.onCreate(savedInstanceState);

        cameraExecutor = Executors.newSingleThreadExecutor();
        copyModelAndInitDnn();

        if (allPermissionsGranted()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, PERMISSION_REQUEST_CODE);
        }
    }

    private void copyModelAndInitDnn() {
        try {
            File out = new File(getFilesDir(), "yolov5n.onnx");
            if (!out.exists() || out.length() < 1_000_000L) {
                try (InputStream is = getAssets().open("dnn/yolov5n.onnx");
                     FileOutputStream fos = new FileOutputStream(out)) {
                    byte[] buf = new byte[16384];
                    int n;
                    while ((n = is.read(buf)) > 0) {
                        fos.write(buf, 0, n);
                    }
                }
            }
            if (!nativeInitDnn(out.getAbsolutePath())) {
                Log.e(TAG, "nativeInitDnn 失败，请检查 ONNX 与 OpenCV 是否匹配");
            }
        } catch (Exception e) {
            Log.e(TAG, "复制或加载 DNN 模型失败（assets 需含 dnn/yolov5n.onnx）", e);
        }
    }

    private ByteBuffer mDirectBuffer;
    private byte[] mRowData; // 用于中转单行数据

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(this);

        cameraProviderFuture.addListener(() -> {
            try {
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();

                int displayRotation = getDisplay().getRotation();

                ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setTargetRotation(displayRotation)
                        .build();

                imageAnalysis.setAnalyzer(cameraExecutor, image -> {
                    ImageProxy.PlaneProxy plane = image.getPlanes()[0];
                    ByteBuffer buffer = plane.getBuffer();
                    
                    int width = image.getWidth();
                    int height = image.getHeight();
                    int rowStride = plane.getRowStride();
                    int rotationDegrees = image.getImageInfo().getRotationDegrees();
                    
                    int requiredSize = width * height * 4;
                    if (mDirectBuffer == null || mDirectBuffer.capacity() != requiredSize) {
                        mDirectBuffer = ByteBuffer.allocateDirect(requiredSize);
                        mRowData = new byte[width * 4];
                    }
                    
                    mDirectBuffer.clear();
                    for (int row = 0; row < height; row++) {
                        buffer.position(row * rowStride);
                        buffer.get(mRowData);
                        mDirectBuffer.put(mRowData);
                    }

                    // 传递直接缓冲区数据和旋转角度到 C++
                    processCameraFrameDirect(mDirectBuffer, width, height, rotationDegrees);
                    
                    image.close();
                });

                CameraSelector cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA;
                cameraProvider.unbindAll();
                cameraProvider.bindToLifecycle(this, cameraSelector, imageAnalysis);

            } catch (ExecutionException | InterruptedException e) {
                Log.e(TAG, "相机绑定失败", e);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private boolean allPermissionsGranted() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_CODE && allPermissionsGranted()) {
            startCamera();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        cameraExecutor.shutdown();
    }
}
