// *********Calibration.cpp***********
// Created by xinzhichao on 2021/6/15.
// Email: xinzhichaotongxue@foxmail.com.
//

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

Size const cornerSize = Size(8, 6);                       //定义标定板每行、每列角点数
Size squareSize = Size(26.5, 26.5);                           //实际测量得到的标定板上每个棋盘格的尺寸，单位mm
String const calibrationFileName = "../calibration/txt/calibrationData.txt"; //标定所用图像文件的路径
String const resultFileName = "../calibration/txt/calibrationResult.txt";    //保存标定结果的文件
String const rectifyFileName = "../calibration/images/before.jpg";          //待校正的图像文件
String const rectifiedFileName = "../calibration/images/rectified.jpg";  //校正后的图像文件

struct calibrationCache
{
    vector<Point2f> cornerPointBuffer;          //缓存每幅图像上检测到的角点
    vector<vector<Point2f>> cornerPointSequece; //保存检测到的所有角点
    vector<vector<Point3f>> cornerPostion;      //保存标定板上角点的三维坐标
    int imageCount = 0;                         //采样图像数量
    int cornerCount = 0;                        //检测到的角点数量
    Size imageSize;                             //输入图像的像素尺寸
    int image_cal_num = 10;                     //参与标定的图片数量＋１

    //----结果部分---------//
    Mat intrinsicMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); //摄像机内参数矩阵
    Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));      //摄像机的5个畸变系数：k1,k2,p1,p2,k3
    vector<Mat> tvecsMat;                                      //每幅图像的旋转向量
    vector<Mat> rvecsMat;                                      //每幅图像的平移向量
};
//----函数声明---------//
void drawIMage(Mat img, vector<Point2f> cornerBuffer);
void outPutCornerInfo(calibrationCache &cache);
void calibrationCamera(calibrationCache &cache);
void caculateErrorAndSaveResult(calibrationCache &cache);
void unDistortRectifyImage(calibrationCache &cache);

int main()
{
    ifstream fin(calibrationFileName);
    if (!fin)
    {
        cerr << "请在有calibrationFileName定义的文件目录下运行此程序" << endl;
        return 1;
    }
    calibrationCache calibCache;
    string imageName;

    cout << "开始读取图像..." << endl;
    while (getline(fin, imageName))
    {
        calibCache.imageCount++;
        cout << "imageCount = " << calibCache.imageCount << endl;
        Mat imageInput = imread(imageName);
        if (calibCache.imageCount == 1) //读入第一张图片时获取图像宽高信息
        {
            calibCache.imageSize.width = imageInput.cols;
            calibCache.imageSize.height = imageInput.rows;
            cout << "image size width  = " << calibCache.imageSize.width << endl;
            cout << "image size height = " << calibCache.imageSize.height << endl;
        }

        if (calibCache.imageCount == calibCache.image_cal_num)
        {   calibCache.imageCount--;
            break;
        }
        cout << "开始提取第" << calibCache.imageCount << "张图像的角点..." << endl;
        if (0 == findChessboardCorners(imageInput, cornerSize, calibCache.cornerPointBuffer))
        {
            cout << "无法提取第" << calibCache.imageCount << "张图像的角点，程序将跳过该图片！" << endl;
            calibCache.imageCount--;
            continue;
        }
        else
        {
            Mat grayImage;
            cvtColor(imageInput, grayImage, CV_RGB2GRAY);

            find4QuadCornerSubpix(grayImage, calibCache.cornerPointBuffer, Size(5, 5)); //对提取的角点进行精细化
            calibCache.cornerPointSequece.push_back(calibCache.cornerPointBuffer);      //保存亚像素角点

            drawIMage(grayImage, calibCache.cornerPointBuffer); //在图像上显示角点位置
        }
    }
    cout << "角点提取完成！" << endl
         << endl;
    outPutCornerInfo(calibCache);

    cout << "开始标定..." << endl;
    calibrationCamera(calibCache);

    cout << "开始评价标定结果..." << endl;
    caculateErrorAndSaveResult(calibCache);

    cout << "验证效果..." << endl;
    unDistortRectifyImage(calibCache);
    return 0;
}

///显示图像
void drawIMage(Mat img, vector<Point2f> cornerBuffer)
{
    drawChessboardCorners(img, cornerSize, cornerBuffer, false); //用于在图片中标记角点
    imshow("Camera Calibration", img);                           //显示图片
    waitKey(1000);                                               //暂停1
}
///输出所有角点信息
void outPutCornerInfo(calibrationCache &cache)
{
    int SampleImageCount = cache.cornerPointSequece.size();
    cout << "采样图像数= " << SampleImageCount << endl;
    if (SampleImageCount < 1)
        exit(1);
    for (int j = 0; j < cache.cornerPointSequece.size(); j++)
    {
        cout << endl;
        cout << "第" << j + 1 << "张图片的角点数据: " << endl;
        for (int i = 0; i < cache.cornerPointSequece[j].size(); i++)
        {
            cout << "(X:" << cache.cornerPointSequece[j][i].x << ",Y:" << cache.cornerPointSequece[j][i].y << ")"
                 << "      ";
            if (0 == (i + 1) % 4) // 格式化输出，便于控制台查看
            {
                cout << endl;
            }
        }
        cout << endl;
    }
    cout << endl;
}
///标定相机
void calibrationCamera(calibrationCache &cache)
{
    int i, j, t;
    for (t = 0; t < cache.imageCount; t++)
    {
        vector<Point3f> tempCornerPosition;
        for (i = 0; i < cornerSize.height; i++)
        {
            for (j = 0; j < cornerSize.width; j++)
            {
                Point3f cornerPos;
                cornerPos.x = i * squareSize.width;
                cornerPos.y = j * squareSize.height;
                cornerPos.z = 0; //假设标定板放在世界坐标系中z=0的平面上
                tempCornerPosition.push_back(cornerPos);
            }
        }
        cache.cornerPostion.push_back(tempCornerPosition);
    }
    //开始标定相机
    calibrateCamera(cache.cornerPostion, cache.cornerPointSequece, cache.imageSize, cache.intrinsicMatrix, cache.distCoeffs, cache.rvecsMat, cache.tvecsMat, 0);
    cout << "标定完成！" << endl
         << endl;
}
//评估误差并保存结果
void caculateErrorAndSaveResult(calibrationCache &cache)
{
    int cornerCount = cornerSize.width * cornerSize.height; //每幅图像中角点数量，假定每幅图像中都可以看到完整的标定板
    double totalError = 0.0;                                //所有图像的平均误差的总和
    double sigleError = 0.0;                                //每幅图像的平均误差
    vector<Point2f> reprojectPoints;                        //保存重新计算得到的投影点
    ofstream fout(resultFileName);
    cout << "每幅图像的标定误差：" << endl;
    fout << "每幅图像的标定误差：\n";
    for (int i = 0; i < cache.imageCount; i++)
    {
        vector<Point3f> tempPointSet = cache.cornerPostion[i];
        // 通过得到的摄像机内外参数，对空间的三维点进行重新投影计算，得到新的投影点
        projectPoints(tempPointSet, cache.rvecsMat[i], cache.tvecsMat[i], cache.intrinsicMatrix, cache.distCoeffs, reprojectPoints);
        // 计算新的投影点和旧的投影点之间的误差
        vector<Point2f> oldImagePoint = cache.cornerPointSequece[i];
        Mat oldImagePointMatrix = Mat(1, oldImagePoint.size(), CV_32FC2);
        Mat reprojectPointsMatrix = Mat(1, reprojectPoints.size(), CV_32FC2);
        for (int j = 0; j < oldImagePoint.size(); j++)
        {
            reprojectPointsMatrix.at<Vec2f>(0, j) = Vec2f(reprojectPoints[j].x, reprojectPoints[j].y);
            oldImagePointMatrix.at<Vec2f>(0, j) = Vec2f(oldImagePoint[j].x, oldImagePoint[j].y);
        }
        sigleError = norm(reprojectPointsMatrix, oldImagePointMatrix, NORM_L2);
        totalError += (sigleError /= cornerCount);
        cout << "第" << i + 1 << "幅图像的平均误差：" << sigleError << "像素" << endl;
        fout << "第" << i + 1 << "幅图像的平均误差：" << sigleError << "像素" << endl;
    }
    cout << "总体平均误差：" << totalError / cache.imageCount << "像素" << endl;
    fout << "总体平均误差：" << totalError / cache.imageCount << "像素" << endl
         << endl;
    cout << "评价完成！" << endl
         << endl;

    cout << "开始保存定标结果..." << endl;
    Mat rotationMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); //保存每幅图像的旋转矩阵
    fout << "相机内参数矩阵：" << endl;
    cout << "相机内参数矩阵：" << endl;
    fout << cache.intrinsicMatrix << endl
         << endl;
    cout << cache.intrinsicMatrix << endl
         << endl;
    fout << "畸变系数：\n";
    cout << "畸变系数：" << endl;
    fout << cache.distCoeffs << endl
         << endl
         << endl;
    cout << cache.distCoeffs << endl
         << endl
         << endl;
    for (int i = 0; i < cache.imageCount; i++)
    {
        fout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
        cout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
        fout << cache.rvecsMat[i] << endl;
        cout << cache.rvecsMat[i] << endl;
        /* 罗德里格斯公式，将旋转向量转换为旋转矩阵 */
        Rodrigues(cache.rvecsMat[i], rotationMatrix);
        fout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
        cout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
        fout << rotationMatrix << endl;
        cout << rotationMatrix << endl;
        fout << "第" << i + 1 << "幅图像的平移向量：" << endl;
        cout << "第" << i + 1 << "幅图像的平移向量：" << endl;
        fout << cache.tvecsMat[i] << endl
             << endl;
        cout << cache.tvecsMat[i] << endl
             << endl;
    }
    cout << "完成保存" << endl
         << endl;
    fout << endl;
}
///校正图像
void unDistortRectifyImage(calibrationCache &cache)
{
    Mat sourceImage = imread(rectifyFileName);
    Mat rectifyImage = sourceImage.clone();
    Size imageSize = Size(sourceImage.cols, sourceImage.rows);
    Mat firstMap = Mat(imageSize, CV_32FC1);
    Mat secondMap = Mat(imageSize, CV_32FC1);
    Mat I = Mat::eye(3, 3, CV_32F);
    cout << "获取校正图..." << endl;
    initUndistortRectifyMap(cache.intrinsicMatrix, cache.distCoeffs, I, cache.intrinsicMatrix, imageSize, CV_32FC1, firstMap, secondMap);

    remap(sourceImage, rectifyImage, firstMap, secondMap, INTER_LINEAR);
    //undistort(sourceImage, rectifyImage, cache.intrinsicMatrix, cache.distCoeffs); //另一种不需要转换矩阵的方式
    imwrite(rectifiedFileName, rectifyImage);
    std::cout << "校正结束" << endl;
    imshow("rectified image", rectifyImage); //显示校正后图片
//    waitKey(1000);                           //暂停1秒
    waitKey(-1);
}


