import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

public class FaceDetectionAndRecognition {

    static {
        // Load the OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        // Load the pre-trained face detection model
        CascadeClassifier faceDetector = new CascadeClassifier("haarcascade_frontalface_default.xml");

        // Load the image
        String imagePath = "input.jpg";
        Mat image = Imgcodecs.imread(imagePath);

        // Detect faces in the image
        MatOfRect faces = new MatOfRect();
        faceDetector.detectMultiScale(image, faces);

        // Draw rectangles around the detected faces
        List<Rect> faceList = faces.toList();
        for (Rect face : faceList) {
            Imgproc.rectangle(image, new Point(face.x, face.y),
                    new Point(face.x + face.width, face.y + face.height), new Scalar(0, 255, 0), 2);
        }

        // Save the output image with detected faces
        Imgcodecs.imwrite("output.jpg", image);

        // Optional: Implement face recognition
        // ...
    }
}