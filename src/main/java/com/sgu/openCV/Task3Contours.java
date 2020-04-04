package com.sgu.openCV;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY;
import static org.opencv.imgproc.Imgproc.THRESH_OTSU;

public class Task3Contours {

    public static void main(String[] args) {
        System.load("/C:/Users/Lunge/opencv/build/java/x64/opencv_java420.dll");

        Mat img = Imgcodecs.imread("inputTask3.jpg");
        if (img.empty()) {
            System.out.println("Не удалось загрузить изображение");
            return;
        }
        Mat imgGray = new Mat();
        Imgproc.cvtColor(img, imgGray, Imgproc.COLOR_BGR2HSV);
        Imgproc.blur(imgGray, imgGray, new Size(7, 7));

        int cols = img.cols();
        int rows = img.rows();

        Mat mat = new Mat(rows, cols, imgGray.type());

        for (int i = 0; i < img.rows(); i++) {
            for (int j = 0; j < img.cols(); j++) {
                mat.put(i, j, imgGray.get(i, j)[0], 0, 0);
            }
        }

        Mat resMat1 = new Mat();

        Imgproc.cvtColor(mat, resMat1, COLOR_BGR2GRAY);

        Mat resMat2 = new Mat();


        Imgproc.threshold(resMat1, resMat2, 0, 255, THRESH_OTSU);

        Imgcodecs.imwrite("OutputTask3.jpg", resMat2);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(resMat2, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        System.out.println("На фото " + contours.size() + " объектов.");
    }
}
