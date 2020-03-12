---
layout: posts
title: CUDA 를 사용하여 효과적으로 컬러 이미지를 흑백 이미지로 바꿔보자!
tags: [CUDA, Computer Vision, OpenCV, Nsight Compute]
---

<br/>
컬러 이미지는 보통 RGB 형태로 표현 되는데, 이미지의 각 픽셀은 (r, g, b) 값의 튜플로 저장 됩니다. 아래의 그림과 같이 이미지 행은 (r g b), (r g b)... 형식으로 나타내어 집니다. 즉 하나의 픽셀은 r, g 그리고 b 의 값이 합쳐져서 나타내어 지게 됩니다.<br/>

<figure style="margin-left: 5em; margin-right: 5em;">
    <img src="../assets/images/color_rgb.png">
    <figcaption>RGB 컬러 형식</figcaption>
</figure>


컬러 이미지를 흑백 이미지로 바꾸기 위해서는 휘도(Luminance)를 계산 해야 하는데 r, g, b 값을 이용하여 다음과 같은 같단한 식으로 계산 할 수 있습니다.

```math
휘도(Luminance) = r * 0.21 + g * 0.27 + b * 0.07
```

그러면 CUDA 를 이용하여 컬러 이미지를 흑백 이미지로 변환해 봅시다.

이미지 로드를 하기 위해 OpenCV 를 사용 하였는데 OpenCV 는 이미지를 로드 할 때 RGB 순서대로 정보를 저장하지 않고 BGR 순서대로 저장합니다 (kernel 에서 휘도를 계산할때 bgr 순서를 사용).

다음과 같이 이미지를 로드 합니다.

```cpp
// read image by using OpenCV
string filePath = "D:/CUDA/image/hs.jpg";

Mat img = imread(filePath, CV_LOAD_IMAGE_COLOR);

if (!img.data) {
	cout << "Image Load Fail" << endl;
	return 0;
}
```

이제 컬러 이미지 받아 흑백 이미지를 반환하는 함수를 작성하여 봅시다.

함수의 순서는 다음과 같습니다.

1. 컬러 이미지와 같은 크기의 이미지 컨테이너를 생성
2. 컬러 이미지와 흑백 이미지 데이터 사이즈 만큼 GPU 에 메모리 할당
3. 컬러 이미지 데이터를 CPU -> GPU 로 복사
4. kernel 실행 (color to gray)
5. 흑백 이미지 데이터를 GPU -> CPU 로 복사
6. 할당 했던 GPU 메모리 해재

아래의 코드는 위의 순서대로 작성한 코드 입니다. kernel 함수는 더 밑에서 설명하도록 하겠습니다.


```cpp
Mat colorToGray(Mat &colorImg) {
    // make gray image container
    const int width = colorImg.cols, height = colorImg.rows;

	Mat grayImg = Mat::zeros(height, width, CV_8UC1);

	const int grayByte = width * height * sizeof(uchar);
	const int bgrByte = grayByte * 3;
	
	uchar *dImg, *dGrayImg;

    // allocate color and gray image memory in GPU
	cudaMalloc(&dImg, bgrByte);
	cudaMalloc(&dGrayImg, grayByte);

    // send color image data to GPU
    cudaMemcpy(dImg, colorImg.data, bgrByte, cudaMemcpyHostToDevice);

    // kernel block and grid size
	dim3 block(32, 32);
	dim3 grid(width / 32 + 1, height / 32 + 1);

    // kernel function
	cuColorToGray<<<grid, block>>>(dGrayImg, dImg, width, height);

    // get gray image data from GPU 
	cudaMemcpy(grayImg.data, dGrayImg, grayByte, cudaMemcpyDeviceToHost);

    // free GPU memory space
	cudaFree(dImg);
	cudaFree(dGrayImg);

	return grayImg;
}
```

block 의 크기는 최대 thread 수인 1024 개를 사용하기 위해 32x32 사이즈를 할당했고 grid 는 block 크기와 이미지 크기에 따라 정하였습니다. 

kernel 함수 에서는 단순이 BGR 순서대로 값을 읽고 위의 식대로 휘도(Luminance) 를 계산하였습니다.

```cpp
// CUDA kernel for converting rgb t0 gray
__global__
void cuColorToGray(uchar *d_out, uchar *d_in, const int width, const int height) {
	const int col = blockIdx.x * blockDim.x + threadIdx.x;
	const int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < width && row < height) {
		uchar b = d_in[row * width * 3 + col * 3];
		uchar g = d_in[row * width * 3 + col * 3 + 1];
		uchar r = d_in[row * width * 3 + col * 3 + 2];

        // luminance
		d_out[row * width + col] = (uchar)(r * 0.299f + g * 0.587f + b * 0.114f);
	}
}
```

작성한 코드로 컬러 이미지를 흑백 이미지로 변경한 결과를 확인 하여 봅시다.

```cpp
const int width = img.cols;
const int height = img.rows;

namedWindow("COLOR_HS", WINDOW_NORMAL);
namedWindow("GRAY_HS", WINDOW_NORMAL);

resizeWindow("COLOR_HS", width, height);
resizeWindow("GRAY_HS", width, height);

imshow("COLOR_HS", img);
imshow("GRAY_HS", grayImg);
```


<figure class="HS">
    <img src="../assets/images/hs.jpg" width="300">
    <img src="../assets/images/GRAY_HS.jpg" width="300">
    <figcaption>컬러 이미지와 흑백 이미지</figcaption>
</figure>



kernel 함수를 직접 작성하지 않고 흑백 변환 함수를 만들 수 있을가요? NVIDIA 에서 제공하는 **NPP(NVIDIA Performance Primitives) library** 를 사용하면 간단합니다.

기존 코드에서 block 과 grid 크기를 선언하고 kernel 을 콜한 부분을 다음과 같은 코드로 변경하면 간단하게 만들 수 있습니다.

```cpp
const int dImgPitch = width * 3;
const int dGrayImgPitch = width;
const NppiSize oSizeROI = { width, height };
const Npp32f aCoeffs[] = { 0.299f, 0.587f, 0.114f };

// npp color to gray converting function
nppiColorToGray_8u_C3C1R(dImg, dImgPitch, dGrayImg, dGrayImgPitch, oSizeROI, aCoeffs);
```

그렇다면 직접 작성한 kernel 함수와 NPP library 의 성능 비교 Nsight Compute를 통해 해 봅시다.

아래의 그림은 직접 작성한 kernel 함수와 NPP library 의 kernel 성능 결과 입니다. Current 가 NPP library 이고 Baseline 1 이 직접 작성한 코드 입니다.

<figure class="HS">
    <img src="../assets/images/performance_result1.png" width="700">
	<img src="../assets/images/performance_result2.png" width="700">
    <figcaption>ColorToGray 와 NPP 결과 성능 비교</figcaption>
</figure>

결과를 보면 알 수 있듯이 직접 작성한  kernel 의 경우가 성능이 더 좋게 나온것을 알 수 있습니다. 

다음번 포스트 에는 cuColorToGray kernel 이 왜 더 성능이 좋게 나왔는지를 Nsight Compute report 를 보고 분석해 보도록 하겠습니다!
