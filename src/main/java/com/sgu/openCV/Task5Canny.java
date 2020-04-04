package com.sgu.openCV;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY;

public class Task5Canny {
    public static void main(String[] args) {
        System.load("/C:/Users/Lunge/opencv/build/java/x64/opencv_java420.dll");

        Mat img = Imgcodecs.imread("inputTask2.png");
        if (img.empty()) {
            System.out.println("Не удалось загрузить изображение");
            return;
        }

        Mat imgGrey = new Mat();
        Imgproc.cvtColor(img, imgGrey, COLOR_BGR2GRAY);
        Mat blurImg = new Mat();

        Imgproc.GaussianBlur(imgGrey, blurImg, new Size(7, 7), 1);
        List<Mat> matList = sobel(blurImg);
        Mat nonMaxMat = nonMaximumSuppression(matList.get(0), matList.get(1));
        Mat thresholdMat = doubleThreshold(nonMaxMat, 0.5, 0.55);
        Mat resMat = tracing(thresholdMat, 255, 0);
        Imgcodecs.imwrite("OutputTask5.jpg", resMat);
    }

    public static List<Mat> sobel(Mat imgMat) {
        double[] kernelXMas = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
        double[] kernelYMas = {1, 2, 1, 0, 0, 0, -1, -2, -1};
        Mat kernelX = new Mat(3, 3, CvType.CV_32FC1);
        kernelX.put(0, 0, kernelXMas);
        Mat kernelY = new Mat(3, 3, CvType.CV_32FC1);
        kernelY.put(0, 0, kernelYMas);
        System.out.println(kernelX.dump());
        System.out.println(kernelY.dump());

        Mat xMat = new Mat();
        Imgproc.filter2D(imgMat, xMat, -1, kernelX);
        Mat yMat = new Mat();
        Imgproc.filter2D(imgMat, yMat, -1, kernelY);

        Mat sobMat = new Mat(imgMat.size(), imgMat.type());
        Mat thetaMat = new Mat(imgMat.size(), imgMat.type());

        for (int i = 0; i < imgMat.rows(); i++) {
            for (int j = 0; j < imgMat.cols(); j++) {
                double dX = xMat.get(i, j)[0];
                double dY = yMat.get(i, j)[0];
                double g = Math.hypot(dX, dY);
                double theta = Math.round(Math.atan2(dX, dY) / (Math.PI / 4)) * (Math.PI / 4) - (Math.PI / 2);
                sobMat.put(i, j, g);
                thetaMat.put(i, j, theta);

            }
        }

        List<Mat> resList = new ArrayList<>(2);
        resList.add(sobMat);
        resList.add(thetaMat);
        return resList;
    }

    private static boolean isUnCorrectIndex(Mat mat, int x, int y) {
        return x < 0 || x >= mat.rows() || y < 0 || y >= mat.cols();
    }

    private static boolean check(Mat mat, int x, int y, double value) {
        if (!isUnCorrectIndex(mat, x, y)) {
            try {
                return mat.get(x, y)[0] <= value;

            } catch (Exception e) {
                System.out.println(" ");
            }
        }
        return false;
    }

    private static Mat nonMaximumSuppression(Mat sobMat, Mat thetaMat) {
        Mat resMat = new Mat(sobMat.size(), sobMat.type());

        for (int i = 0; i < sobMat.rows(); i++) {
            for (int j = 0; j < sobMat.cols(); j++) {
                int dx = (int) Math.signum(Math.cos(thetaMat.get(i, j)[0]));
                int dy = (int) -Math.signum(Math.sin(thetaMat.get(i, j)[0]));
                if (check(sobMat, i + dx, j + dy, sobMat.get(i, j)[0])) {
                    resMat.put(i + dx, j + dy, 0);
                }
                if (check(sobMat, i - dx, j - dy, sobMat.get(i, j)[0])) {
                    resMat.put(i - dx, j - dy, 0);
                }
                resMat.put(i, j, sobMat.get(i, j)[0]);
            }
        }
        return resMat;
    }

    private static Mat doubleThreshold(Mat mat, double lowPr, double highPr) {
        double down = lowPr * 255;
        double up = highPr * 255;

        Mat resMat = new Mat(mat.size(), mat.type());
        for (int i = 0; i < mat.rows(); i++) {
            for (int j = 0; j < mat.cols(); j++) {
                if (mat.get(i, j)[0] >= up) {
                    resMat.put(i, j, 255);
                } else {
                    if (mat.get(i, j)[0] <= down) {
                        resMat.put(i, j, 0);
                    } else {
                        resMat.put(i, j, 127);
                    }
                }
            }
        }
        return resMat;
    }

    private static Mat tracing(Mat mat, int high, int clear) {
        double[] moveDirArray = {-1, -1, -1, 0, 0, 1, 1, 1, -1, 0, 1, -1, 1, -1, 0, 1};
        Mat moveDirMat = new Mat(2, 8, mat.type());
        moveDirMat.put(0, 0, moveDirArray);

        Mat resMat = new Mat(mat.size(), mat.type());
        for (int i = 0; i < mat.rows(); i++) {
            for (int j = 0; j < mat.cols(); j++) {
                if (Double.compare(mat.get(i, j)[0], high) == 0) {
                    resMat.put(i, j, high);
                    for (int k = 0; k < 8; k++) {
                        int dx = (int) moveDirMat.get(0, k)[0];
                        int dy = (int) moveDirMat.get(1, k)[0];
                        int x = dx;
                        int y = dy;
                        while (true) {
                            x += dx;
                            y += dy;
                            if (x < 0 || y < 0 || x >= mat.rows() || y >= mat.cols()) {
                                break;
                            }
                            if (mat.get(x, y)[0] == high || mat.get(x, y)[0] == clear) {
                                break;
                            }
                            resMat.put(x, y, high);
                        }
                    }
                } else {
                    resMat.put(i, j, clear);
                }
            }
        }
        return resMat;
    }
}
