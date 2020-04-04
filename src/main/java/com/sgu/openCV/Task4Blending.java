package com.sgu.openCV;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.opencv.imgproc.Imgproc.LINE_4;

public class Task4Blending {

    private static int EYE_X = 670;
    private static int EYE_Y = 230;
    private static int EYE_WIDTH = 110;
    private static int EYE_HEIGHT = 60;
    private static int MASK_X = 740;
    private static int MASK_Y = 170;
    private static int DEPTH = 4;
    private static int FRAME = 40;
    private static int MASK_PARAMETER = 1;

    public static void main(String[] args) {
        System.load("/C:/Users/Lunge/opencv/build/java/x64/opencv_java420.dll");

        Mat inputImg = Imgcodecs.imread("InputTask4.jpg");
        Mat eye_mask = getEyeMask(inputImg);
        Imgcodecs.imwrite("OutputTask4MaskEye.jpg", eye_mask);
        Mat mask = getMask(inputImg);
        Imgcodecs.imwrite("OutputTask4Mask.jpg", mask);

        List<Mat> gaussInput = generateGaussianPyr(inputImg);
        List<Mat> gaussEye = generateGaussianPyr(eye_mask);
        List<Mat> gaussMask = generateGaussianPyr(mask);

        gaussMask.remove(gaussMask.size() - 1);
        Collections.reverse(gaussMask);

        List<Mat> laplEye = generateLaplacianPyr(gaussEye);
        List<Mat> laplInput = generateLaplacianPyr(gaussInput);


        List<Mat> resultMatList = new ArrayList<>();
        for (int i = 0; i < DEPTH; i++) {
            Mat inputLap = laplInput.get(i);
            Mat eyeLap = laplEye.get(i);
            Mat maskGauss = gaussMask.get(i);

            Mat subMat = new Mat();
            Mat onesMat = new Mat(inputLap.size(), maskGauss.type());
            onesMat.setTo(new Scalar(MASK_PARAMETER, MASK_PARAMETER, MASK_PARAMETER));
            Core.subtract(onesMat, maskGauss, subMat);

            Mat sumMat = new Mat();
            Core.add(eyeLap.mul(maskGauss), inputLap.mul(subMat), sumMat);
            resultMatList.add(sumMat);
        }

        Mat result = getResultMat(resultMatList);

        Imgcodecs.imwrite("OutputTask4.jpg", result);
    }

    private static Mat getResultMat(List<Mat> matList) {
        Mat res = matList.get(0);
        for (int i = 1; i < DEPTH; i++) {
            Imgproc.pyrUp(res, res, matList.get(i).size());
            Core.add(res, matList.get(i), res);
        }
        return res;
    }

    private static List<Mat> generateGaussianPyr(Mat img) {
        List<Mat> matList = new ArrayList<>();
        matList.add(img);
        Mat pyrMat = img.clone();
        for (int i = 0; i < DEPTH; i++) {
            Mat mat = new Mat();
            Imgproc.pyrDown(pyrMat, mat);
            matList.add(mat);
            pyrMat = mat.clone();
        }
        return matList;
    }

    private static List<Mat> generateLaplacianPyr(List<Mat> gaussianPyr) {
        List<Mat> laplList = new ArrayList<>();
        laplList.add(gaussianPyr.get(DEPTH - 1));

        for (int i = DEPTH - 1; i > 0; i--) {
            Size size = gaussianPyr.get(i - 1).size();
            Mat mat = new Mat();
            Imgproc.pyrUp(gaussianPyr.get(i), mat, size);
            Mat pyrMat = new Mat();
            Core.subtract(gaussianPyr.get(i - 1), mat, pyrMat);
            laplList.add(pyrMat);
        }
        return laplList;
    }

    private static Mat getMask(Mat inputOne) {
        Mat mask = Mat.zeros(inputOne.size(), inputOne.type());

        Point poly[] = new Point[4];
        poly[0] = new Point(MASK_X, MASK_Y);
        poly[1] = new Point(MASK_X, MASK_Y + EYE_HEIGHT);
        poly[2] = new Point(MASK_X + EYE_WIDTH, MASK_Y + EYE_HEIGHT);
        poly[3] = new Point(MASK_X + EYE_WIDTH, MASK_Y);

        MatOfPoint point = new MatOfPoint(poly);
        List<MatOfPoint> list = Collections.singletonList(point);

        Imgproc.fillPoly(mask, list, new Scalar(MASK_PARAMETER, MASK_PARAMETER, MASK_PARAMETER), LINE_4, 0, new Point(0, 0));

        return mask;
    }

    private static Mat getEyeMask(Mat inputOne) {
        Mat mask = Mat.zeros(inputOne.size(), inputOne.type());
        int halfFrame = FRAME / 2;
        int height = EYE_HEIGHT + FRAME;
        int width = EYE_WIDTH + FRAME;
        Mat eye = new Mat(height, width, inputOne.type());
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                eye.put(i, j, inputOne.get(i + EYE_Y - halfFrame, j + EYE_X - halfFrame));
            }
        }
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                mask.put(i + MASK_Y - halfFrame, j + MASK_X - halfFrame, eye.get(i, j));
            }
        }
        return mask;
    }
}
