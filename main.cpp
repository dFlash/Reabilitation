#include<opencv2/highgui/highgui.hpp>
#include<opencv/cvaux.h>


#include <opencv/cv.h>
#include <opencv/highgui.h>

#include<cmath>
#include<QTime>
#include"gui.hpp"

using namespace std;


//код для трекинга
int currLeftHandY=175,currLeftHandX=50;
int currRightHandY = 131,currRightHandX=434;

int currLeftElbowY=213,currLeftElbowX=148;
int currRightElbowY = 197,currRightElbowX=370;

int leftSize=0, rightSize=0;

const int step = 50;

bool firstIter = true;

int moveL[8][2]={
//                 {-1, 1},
//                 {0, 1},
//                 {1, 1},
//                 {1, 0},
//                 {1, -1},
//                 {0, -1},
//                 {-1, -1},
//                 {-1, 0}
//                };

{1, -1},
{1, 0},
{1, 1},
{0, 1},
{-1, 1},
{-1, 0},
{-1, -1},
{0, -1}
};

int moveR[8][2]={
    {1, 1},
    {1, 0},
    {1, -1},
    {0, -1},
    {-1, -1},
    {-1, 0},
    {-1, 1},
    {0, 1}

    };
inline void getForearmSize(cv::Mat& img,bool isFirst);

inline void initTrack(cv::Mat& gray, bool isFirst)
{
    //левая ладонь

    for(int i = currLeftHandY-step; i < currLeftHandY+step; i++)
    {
        for(int j = currLeftHandX-step; j < currLeftHandX+step; j++)
        {
            if (gray.at<uchar>(i,j)!=255)
            {
                continue;
            }
            else
            {
                uchar p2 = gray.at<uchar>(i-1, j);
                uchar p3 = gray.at<uchar>(i-1, j+1);
                uchar p4 = gray.at<uchar>(i, j+1);
                uchar p5 = gray.at<uchar>(i+1, j+1);

                uchar p6 = gray.at<uchar>(i+1, j);
                uchar p7 = gray.at<uchar>(i+1, j-1);
                uchar p8 = gray.at<uchar>(i, j-1);
                uchar p9 = gray.at<uchar>(i-1, j-1);

                if ((p2+p3+p4+p5+p6+p7+p8+p9)==255)
                {
                    currLeftHandY = i;
                    currLeftHandX = j;
                    break;
                }
            }
        }
    }

    //правая ладонь

    for(int i = currRightHandY-step; i < currRightHandY+step; i++)
    {
        for(int j = currRightHandX-step; j < currRightHandX+step; j++)
        {
            if (gray.at<uchar>(i,j)!=255)
            {
                continue;
            }
            else
            {
                uchar p2 = gray.at<uchar>(i-1, j);
                uchar p3 = gray.at<uchar>(i-1, j+1);
                uchar p4 = gray.at<uchar>(i, j+1);
                uchar p5 = gray.at<uchar>(i+1, j+1);

                uchar p6 = gray.at<uchar>(i+1, j);
                uchar p7 = gray.at<uchar>(i+1, j-1);
                uchar p8 = gray.at<uchar>(i, j-1);
                uchar p9 = gray.at<uchar>(i-1, j-1);

                if ((p2+p3+p4+p5+p6+p7+p8+p9)==255)
                {
                    currRightHandY = i;
                    currRightHandX = j;
                    break;
                }
            }
        }
    }

    if (isFirst)
    {

    //левый локоть
    int elbowLeftX = 0, elbowLeftY = 0;
    for(int i = currLeftElbowY-step; i < currLeftElbowY+step; i++)
    {
        for(int j = currLeftElbowX-step; j < currLeftElbowX+step; j++)
        {
            if (gray.at<uchar>(i,j)!=255)
            {
                continue;
            }
            else
            {
                if (i>elbowLeftY)
                {
                    elbowLeftY=i;
                    elbowLeftX=j;
                }
            }
        }
    }

    currLeftElbowY=elbowLeftY;
    currLeftElbowX=elbowLeftX;


    //правый локоть
    int elbowRightX = 0, elbowRightY = 0;
    for(int i = currRightElbowY-step; i < currRightElbowY+step; i++)
    {
        for(int j = currRightElbowX-step; j < currRightElbowX+step; j++)
        {
            if (gray.at<uchar>(i,j)!=255)
            {
                continue;
            }
            else
            {
                if (i>elbowRightY)
                {
                    elbowRightY=i;
                    elbowRightX=j;
                }
            }
        }
    }

    currRightElbowY=elbowRightY;
    currRightElbowX=elbowRightX;


    getForearmSize(gray,true);

    }//end_if
    else
    {
        getForearmSize(gray,false);
    }




}

inline void getForearmSize(cv::Mat& img,bool isFirst)
{
    cv::Mat tempImg;
    img.copyTo(tempImg);

    int tempLX = currLeftHandX, tempLY = currLeftHandY;

    int tempRX = currRightHandX, tempRY = currRightHandY;

    if (isFirst)
    {
    while ((tempLX != currLeftElbowX) || (tempLY != currLeftElbowY))
    {
        for (int i=0;i<8;i++)
        {
            if(tempImg.at<uchar>(tempLY+moveL[i][0],tempLX+moveL[i][1])==255)
            {
                tempImg.at<uchar>(tempLY,tempLX)=0;
                leftSize++;
                tempLX+=moveL[i][1];
                tempLY+=moveL[i][0];


                break;
            }
        }


    }

    while ((tempRX != currRightElbowX) || (tempRY != currRightElbowY))
    {
        for (int i=0;i<8;i++)
        {
            if(tempImg.at<uchar>(tempRY+moveR[i][0],tempRX+moveR[i][1])!=0)
            {
                tempImg.at<uchar>(tempRY,tempRX)=0;
                tempRX+=moveR[i][1];
                tempRY+=moveR[i][0];
                rightSize++;
                break;
            }
        }

        qDebug()<<"tempRX = "<<tempRX<<" tempRY = "<<tempRY;
        qDebug()<<"currRightHandX = "<<currRightHandX<<" currRightHandY = "<<currRightHandY;
        qDebug()<<"currRightElbowX = "<<currRightElbowX<<" currRightElbowY = "<<currRightElbowY;

    }



    }
    else
    {
        for (int i = 0; i < leftSize; i++)
        {
            for (int i=0;i<8;i++)
            {
                if(tempImg.at<uchar>(tempLY+moveL[i][0],tempLX+moveL[i][1])==255)
                {
                    tempImg.at<uchar>(tempLY,tempLX)=0;
                    tempLX+=moveL[i][1];
                    tempLY+=moveL[i][0];
                    break;
                }
            }
        }

        currLeftElbowY=tempLY;
        currLeftElbowX=tempLX;


        for (int i = 0; i < rightSize; i++)
        {
            for (int i=0;i<8;i++)
            {
                if(tempImg.at<uchar>(tempRY+moveR[i][0],tempRX+moveR[i][1])!=0)
                {
                    //tempImg.at<uchar>(tempRY,tempRX)=0;
                    tempRX+=moveR[i][1];
                    tempRY+=moveR[i][0];
                    break;
                }
            }
        }

        currRightElbowY=tempRY;
        currRightElbowX=tempRX;
    }

}

//код для трекинга КОНЕЦ

const short threshold =35;//80 for test 6  110 for test 3
cv::Rect roi(1,1,630,312);
inline void setForeground(cv::Mat& back, cv::Mat& curr, cv::Mat& fore);
inline void getSkeleton(cv::Mat& fore);
inline void StentifordThinning(cv::Mat& fore);


//японский код

void myThinningInit(CvMat** kpw, CvMat** kpb)
{
  //cvFilter2D用のカーネル
  //アルゴリズムでは白、黒のマッチングとなっているのをkpwカーネルと二値画像、
  //kpbカーネルと反転した二値画像の2組に分けて畳み込み、その後でANDをとる
  for (int i=0; i<8; i++){
    *(kpw+i) = cvCreateMat(3, 3, CV_8UC1);
    *(kpb+i) = cvCreateMat(3, 3, CV_8UC1);
    cvSet(*(kpw+i), cvRealScalar(0), NULL);
    cvSet(*(kpb+i), cvRealScalar(0), NULL);
  }
  //cvSet2Dはy,x(row,column)の順となっている点に注意
  //kernel1
  cvSet2D(*(kpb+0), 0, 0, cvRealScalar(1));
  cvSet2D(*(kpb+0), 0, 1, cvRealScalar(1));
  cvSet2D(*(kpb+0), 1, 0, cvRealScalar(1));
  cvSet2D(*(kpw+0), 1, 1, cvRealScalar(1));
  cvSet2D(*(kpw+0), 1, 2, cvRealScalar(1));
  cvSet2D(*(kpw+0), 2, 1, cvRealScalar(1));
  //kernel2
  cvSet2D(*(kpb+1), 0, 0, cvRealScalar(1));
  cvSet2D(*(kpb+1), 0, 1, cvRealScalar(1));
  cvSet2D(*(kpb+1), 0, 2, cvRealScalar(1));
  cvSet2D(*(kpw+1), 1, 1, cvRealScalar(1));
  cvSet2D(*(kpw+1), 2, 0, cvRealScalar(1));
  cvSet2D(*(kpw+1), 2, 1, cvRealScalar(1));
  //kernel3
  cvSet2D(*(kpb+2), 0, 1, cvRealScalar(1));
  cvSet2D(*(kpb+2), 0, 2, cvRealScalar(1));
  cvSet2D(*(kpb+2), 1, 2, cvRealScalar(1));
  cvSet2D(*(kpw+2), 1, 0, cvRealScalar(1));
  cvSet2D(*(kpw+2), 1, 1, cvRealScalar(1));
  cvSet2D(*(kpw+2), 2, 1, cvRealScalar(1));
  //kernel4
  cvSet2D(*(kpb+3), 0, 2, cvRealScalar(1));
  cvSet2D(*(kpb+3), 1, 2, cvRealScalar(1));
  cvSet2D(*(kpb+3), 2, 2, cvRealScalar(1));
  cvSet2D(*(kpw+3), 0, 0, cvRealScalar(1));
  cvSet2D(*(kpw+3), 1, 0, cvRealScalar(1));
  cvSet2D(*(kpw+3), 1, 1, cvRealScalar(1));
  //kernel5
  cvSet2D(*(kpb+4), 1, 2, cvRealScalar(1));
  cvSet2D(*(kpb+4), 2, 2, cvRealScalar(1));
  cvSet2D(*(kpb+4), 2, 1, cvRealScalar(1));
  cvSet2D(*(kpw+4), 0, 1, cvRealScalar(1));
  cvSet2D(*(kpw+4), 1, 1, cvRealScalar(1));
  cvSet2D(*(kpw+4), 1, 0, cvRealScalar(1));
  //kernel6
  cvSet2D(*(kpb+5), 2, 0, cvRealScalar(1));
  cvSet2D(*(kpb+5), 2, 1, cvRealScalar(1));
  cvSet2D(*(kpb+5), 2, 2, cvRealScalar(1));
  cvSet2D(*(kpw+5), 0, 2, cvRealScalar(1));
  cvSet2D(*(kpw+5), 0, 1, cvRealScalar(1));
  cvSet2D(*(kpw+5), 1, 1, cvRealScalar(1));
  //kernel7
  cvSet2D(*(kpb+6), 1, 0, cvRealScalar(1));
  cvSet2D(*(kpb+6), 2, 0, cvRealScalar(1));
  cvSet2D(*(kpb+6), 2, 1, cvRealScalar(1));
  cvSet2D(*(kpw+6), 0, 1, cvRealScalar(1));
  cvSet2D(*(kpw+6), 1, 1, cvRealScalar(1));
  cvSet2D(*(kpw+6), 1, 2, cvRealScalar(1));
  //kernel8
  cvSet2D(*(kpb+7), 0, 0, cvRealScalar(1));
  cvSet2D(*(kpb+7), 1, 0, cvRealScalar(1));
  cvSet2D(*(kpb+7), 2, 0, cvRealScalar(1));
  cvSet2D(*(kpw+7), 1, 1, cvRealScalar(1));
  cvSet2D(*(kpw+7), 1, 2, cvRealScalar(1));
  cvSet2D(*(kpw+7), 2, 2, cvRealScalar(1));
}

//японский код _КОНЕЦ




inline cv::Mat morphSkeleton(cv::Mat& img);

struct Point
{
    int x;
    int y;

    Point(int X, int Y)
    {
        x=X;
        y=Y;
    }
};

vector<Point> pointToDel;


//not my func BEGIN
void thinningIteration(cv::Mat& im, int iter)
{
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows-1; i++)
    {
        for (int j = 1; j < im.cols-1; j++)
        {
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);

            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);
            uchar p9 = im.at<uchar>(i-1, j-1);

            int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                     (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                     (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                     (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
            int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                marker.at<uchar>(i,j) = 1;
        }
    }

    im &= ~marker;
}

void thinning(cv::Mat& im)
{
    im /= 255;

    cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningIteration(im, 0);
        thinningIteration(im, 1);
        cv::absdiff(im, prev, diff);
        im.copyTo(prev);
    }
    while (cv::countNonZero(diff) > 0);

    im *= 255;
}

void thinningGuoHallIteration(cv::Mat& im, int iter)
{
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows; i++)
    {
        for (int j = 1; j < im.cols; j++)
        {
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);
            uchar p9 = im.at<uchar>(i-1, j-1);

            int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
                     (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
            int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
            int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
            int N  = N1 < N2 ? N1 : N2;
            int m  = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

            if (C == 1 && (N >= 2 && N <= 3) & m == 0)
                marker.at<uchar>(i,j) = 1;
        }
    }

    im &= ~marker;
}

void thinningGuoHall(cv::Mat& im)
{
    im /= 255;

    cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningGuoHallIteration(im, 0);
        thinningGuoHallIteration(im, 1);
        cv::absdiff(im, prev, diff);
        im.copyTo(prev);
    }
    while (cv::countNonZero(diff) > 0);

    im *= 255;
}


//not my func END


inline void featuresDetect(cv::Mat& skeletonImg);



int main(int argc, char* argv[])
{

    QApplication app(argc,argv);

    CvCapture* capture = NULL;


    //capture = cvCaptureFromCAM(0);
    capture = cvCaptureFromFile("test7.webm");


    cv::Mat curr_frame,background,testImg(480,640,CV_8U);
    cv::Mat foreground(480,640,CV_8U);

    GUI gui;
    gui.show();

    cv::namedWindow("Camera");

    //cv::namedWindow("Fore");

    cv::moveWindow("Camera",200,10);

//    cv::BackgroundSubtractorMOG2 bgsub;
//    std::vector<std::vector<cv::Point> > contours;

    cv::Mat fore_roi;

 //японский
    CvMat** kpb = new CvMat *[8];
    CvMat** kpw = new CvMat *[8];
    myThinningInit(kpw, kpb);

     IplImage src;

 //японский  код

     unsigned int count = 1;


        while(true)
    {

            if (count%2==0)
            {
                count++;
                continue;
            }

            count++;

        curr_frame = cvQueryFrame(capture);



         //японский

         //японский

        if (gui.flagNewBack)
        {
            curr_frame.copyTo(background);

            gui.flagNewBack = false;

            //qDebug()<<background.data[640*480+25];

        }
        if (gui.flag)
        {

             //       thinning(foreground);
            //getSkeleton(foreground);

//             StentifordThinning(foreground);
//                      getSkeleton(foreground);
//              cv::imshow("Ros",foreground);

            //cv::imshow("Fore",foreground);
//            cv::imshow("Background",background);

        }

        if(gui.flagTestImg)
        {
            setForeground(background,curr_frame,foreground);
            cv::erode(foreground,foreground,cv::Mat());
            cv::erode(foreground,foreground,cv::Mat());
            cv::erode(foreground,foreground,cv::Mat());
            cv::erode(foreground,foreground,cv::Mat());
            cv::erode(foreground,foreground,cv::Mat());

            cv::dilate(foreground,foreground,cv::Mat());
            cv::dilate(foreground,foreground,cv::Mat());
            cv::dilate(foreground,foreground,cv::Mat());
            cv::dilate(foreground,foreground,cv::Mat());
            cv::dilate(foreground,foreground,cv::Mat());
            cv::dilate(foreground,foreground,cv::Mat());
            cv::dilate(foreground,foreground,cv::Mat());

            fore_roi = foreground(roi);
            cv::imshow("ROI",fore_roi);

            thinning(fore_roi);


            initTrack(fore_roi,firstIter);
            firstIter = false;


            cv::imshow("Zhang-Suen",fore_roi);

            cv::waitKey(3000);

            cv::ellipse(curr_frame,cv::Point(currLeftHandX,currLeftHandY),cv::Size(10,10),100,0,360,cv::Scalar(255,0,0));
            cv::ellipse(curr_frame,cv::Point(currRightHandX,currRightHandY),cv::Size(10,10),100,0,360,cv::Scalar(255,0,0));

            cv::ellipse(curr_frame,cv::Point(currLeftElbowX,currLeftElbowY),cv::Size(10,10),100,0,360,cv::Scalar(255,0,0));
            cv::ellipse(curr_frame,cv::Point(currRightElbowX,currRightElbowY),cv::Size(10,10),100,0,360,cv::Scalar(255,0,0));




            //fore_roi.copyTo(testImg);
            //gui.flagTestImg=false;
            //break;
        }

        cv::imshow("Camera",curr_frame);

 //       cv::imshow("test",testImg);

        char key = cvWaitKey(1);
        if (key==27)break;

//        bgsub.operator ()(curr_frame,fore);
//        cv::erode(fore,fore,cv::Mat());
//        cv::dilate(fore,fore,cv::Mat());

//        cv::findContours(fore,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
//        cv::drawContours(curr_frame,contours,-1,cv::Scalar(0,0,255),2);
//        cv::imshow("F",fore);
//this is cool=)

    }

        cv::Mat ZhangSuen,GuoHall,Rosenfeld,Stentiford, morph,morphSkel;

        testImg.copyTo(ZhangSuen);
        testImg.copyTo(GuoHall);
        testImg.copyTo(Rosenfeld);
        testImg.copyTo(Stentiford);
        testImg.copyTo(morph);

          QTime time;

        //morphology work
        time.start();
        morphSkel = morphSkeleton(morph);
        qDebug()<<"Morphology worked = "<<time.elapsed();
        cv::imshow("morphology",morphSkel);

        //Rosenfeld work
        time.start();
        getSkeleton(Rosenfeld);
        qDebug()<<"Rosenfeld worked = "<<time.elapsed();
        cv::imshow("Rosenfeld",Rosenfeld);

        //Zhang-Suen work
        time.start();
        thinning(ZhangSuen);
        qDebug()<<"Zhang-Suen worked = "<<time.elapsed();
        cv::imshow("Zhang-Suen",ZhangSuen);

        //Guo-Hall work
        time.start();
        thinningGuoHall(GuoHall);
        qDebug()<<"Guo-Hall worked = "<<time.elapsed();
        cv::imshow("Guo-Hall",GuoHall);

        //Stentiford work
        time.start();
        StentifordThinning(Stentiford);
        qDebug()<<"Stentiford worked = "<<time.elapsed();
        cv::imshow("Stentiford",Stentiford);

        //японский


        src = fore_roi.operator IplImage();


        IplImage* dst = cvCloneImage(&src);

        IplImage* src_w = cvCreateImage(cvGetSize(&src), IPL_DEPTH_32F, 1);
        IplImage* src_b = cvCreateImage(cvGetSize(&src), IPL_DEPTH_32F, 1);
        IplImage* src_f = cvCreateImage(cvGetSize(&src), IPL_DEPTH_32F, 1);
        cvScale(&src, src_f, 1/255.0, 0);

        cvThreshold(src_f,src_f,0.5,1.0,CV_THRESH_BINARY);
        cvThreshold(src_f,src_w,0.5,1.0,CV_THRESH_BINARY);
        cvThreshold(src_f,src_b,0.5,1.0,CV_THRESH_BINARY_INV);

        cvNamedWindow("1");
        cvShowImage("1",src_f);

        double sum=1;

        time.start();

        while(sum>0){
          sum=0;
          for (int i=0; i<8; i++){
            cvFilter2D(src_w, src_w, *(kpw+i));
            cvFilter2D(src_b, src_b, *(kpb+i));

            cvThreshold(src_w,src_w,2.99,1,CV_THRESH_BINARY);
            cvThreshold(src_b,src_b,2.99,1,CV_THRESH_BINARY);
            cvAnd(src_w, src_b, src_w);

            sum += cvSum(src_w).val[0];

            cvXor(src_f, src_w, src_f);

            cvCopyImage(src_f, src_w);
            cvThreshold(src_f,src_b,0.5,1,CV_THRESH_BINARY_INV);
          }
        }

         qDebug()<<"Japaneese worked = "<<time.elapsed();

        cvConvertScaleAbs(src_f, dst, 255, 0);



        cvNamedWindow("dst",1);
        cvShowImage("dst", dst);

        //японский

        cv::waitKey(900000);


    cvReleaseCapture(&capture);
    cvDestroyAllWindows();

    return app.exec();
}

inline void setForeground(cv::Mat& back, cv::Mat& curr, cv::Mat& fore)
{

    for (int i=1;i<back.rows-1;i++)
      {
          for (int j=1;j<back.cols-1;j++)
          {

              if (abs((int)back.at<cv::Vec3b>(i,j)[0]-(int)curr.at<cv::Vec3b>(i,j)[0])>threshold ||
                  abs((int)back.at<cv::Vec3b>(i,j)[1]-(int)curr.at<cv::Vec3b>(i,j)[1])>threshold ||
                  abs((int)back.at<cv::Vec3b>(i,j)[2]-(int)curr.at<cv::Vec3b>(i,j)[2])>threshold
                  )
              {
                  fore.at<unsigned char>(i,j) = 255;

              }
              else
              {
                 fore.at<unsigned char>(i,j) = 0;
              }

          }
      }

}

inline void getSkeleton(cv::Mat& fore)
{

    fore /= 255;
    int dir = 0;
    bool isDel=true;
    while(isDel)
    {

        isDel = false;
        switch(dir%4)
        {
        //сверху вниз
        case 0:

            for (int i = 1; i < fore.cols-1; i++)
            {

                for (int j = 1; j < fore.rows - 1; j++)
                {


                    if (fore.at<unsigned char>(j,i)==0)
                    {
                        continue;
                    }
                    else
                    {
                        uchar p2 = fore.at<uchar>(j-1, i);
                        uchar p3 = fore.at<uchar>(j-1, i+1);
                        uchar p4 = fore.at<uchar>(j, i+1);
                        uchar p5 = fore.at<uchar>(j+1, i+1);

                        uchar p6 = fore.at<uchar>(j+1, i);
                        uchar p7 = fore.at<uchar>(j+1, i-1);
                        uchar p8 = fore.at<uchar>(j, i-1);
                        uchar p9 = fore.at<uchar>(j-1, i-1);

//                        int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
//                                 (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
//                                 (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
//                                 (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);


                        uchar num = p2+p3+p4+p5+p6+p7+p8+p9;
                        if (num > 2 || num==0)
                        {
                            isDel = true;



                           // if (!p3 && p8 || !p9 && p4 || p4 && p8){
                            fore.at<unsigned char>(j,i)=0;
                            break;//}
                        }
                        else
                        {
                            continue;
                        }

                    }
                }
            }

            break;


        //снизу вверх
        case 1:

            for (int i = 1; i < fore.cols-1; i++)
            {

                for (int j = fore.rows-1; j > 1 ; j--)
                {

                    if (fore.at<unsigned char>(j,i)==0)
                    {
                        continue;
                    }
                    else
                    {
                        uchar p2 = fore.at<uchar>(j-1, i);
                        uchar p3 = fore.at<uchar>(j-1, i+1);
                        uchar p4 = fore.at<uchar>(j, i+1);
                        uchar p5 = fore.at<uchar>(j+1, i+1);

                        uchar p6 = fore.at<uchar>(j+1, i);
                        uchar p7 = fore.at<uchar>(j+1, i-1);
                        uchar p8 = fore.at<uchar>(j, i-1);
                        uchar p9 = fore.at<uchar>(j-1, i-1);


                        //                        int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                        //                                 (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                        //                                 (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                        //                                 (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);


                        uchar num = p2+p3+p4+p5+p6+p7+p8+p9;
                        if (num > 2 || num==0)
                        {
                           isDel = true;


                          //if (!p7 && p4 || !p5 && p8 || p8 && p4){
                           fore.at<unsigned char>(j,i)=0;
                            break;//}
                        }
                        else
                        {
                            continue;
                        }
                    }
                }
            }

            break;

        //слева направо
        case 2:

            for (int i = 1; i < fore.rows-1; i++)
            {

                for (int j = 1; j < fore.cols - 1; j++)
                {

                    if (fore.at<unsigned char>(i,j)==0)
                    {
                        continue;
                    }
                    else
                    {
                        uchar p2 = fore.at<uchar>(i-1, j);
                        uchar p3 = fore.at<uchar>(i-1, j+1);
                        uchar p4 = fore.at<uchar>(i, j+1);
                        uchar p5 = fore.at<uchar>(i+1, j+1);

                        uchar p6 = fore.at<uchar>(i+1, j);
                        uchar p7 = fore.at<uchar>(i+1, j-1);
                        uchar p8 = fore.at<uchar>(i, j-1);
                        uchar p9 = fore.at<uchar>(i-1, j-1);

                        //                        int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                        //                                 (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                        //                                 (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                        //                                 (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);


                        uchar num = p2+p3+p4+p5+p6+p7+p8+p9;
                        if (num > 2 || num==0)
                        {
                            isDel = true;

                           // if (!p9 && p6 || !p7 && p2 || p2 && p6){
                            fore.at<unsigned char>(i,j)=0;
                            break;//}
                        }
                        else
                        {
                            continue;
                        }
                    }
                }
            }

            break;

            //справа налево
        case 3:

            for (int i = 1; i < fore.rows-1; i++)
            {

                for (int j = fore.cols - 1; j > 1; j--)
                {

                    if (fore.at<unsigned char>(i,j)==0)
                    {
                        continue;
                    }
                    else
                    {
                        uchar p2 = fore.at<uchar>(i-1, j);
                        uchar p3 = fore.at<uchar>(i-1, j+1);
                        uchar p4 = fore.at<uchar>(i, j+1);
                        uchar p5 = fore.at<uchar>(i+1, j+1);

                        uchar p6 = fore.at<uchar>(i+1, j);
                        uchar p7 = fore.at<uchar>(i+1, j-1);
                        uchar p8 = fore.at<uchar>(i, j-1);
                        uchar p9 = fore.at<uchar>(i-1, j-1);






                        //                        int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                        //                                 (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                        //                                 (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                        //                                 (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);


                        uchar num = p2+p3+p4+p5+p6+p7+p8+p9;
                        if (num > 2 || num==0)
                        {
                            isDel = true;

                            //if (!p5 && p2 || !p3 && p6 || p6 && p2){
                            fore.at<unsigned char>(i,j)=0;
                            break;//}
                        }
                        else
                        {
                            continue;
                        }
                    }
                }
            }
            break;

        }
        dir++;
    }

    fore *= 255;

}

#define V fore.at<uchar>

inline void StentifordThinning(cv::Mat& fore)
{
    fore /= 255;

    bool isFirst = true;

  while (!pointToDel.empty() || isFirst)
  {

      pointToDel.clear();

      //mask1
    for (int i = 2; i < fore.rows-2; i++)
    {
        for (int j = 2; j < fore.cols-2; j++)
        {


            if (V(i,j)==1 && V(i+1,j)==1 && V(i-1,j)==0)
            {

                if (V(i+1,j+1)==0 &&
                    V(i+1,j-1)==0 &&
                    V(i,j+1)==0 &&
                    V(i,j-1)==0 &&
                    V(i-1,j+1)==0 &&
                    V(i-1,j-1)==0 )
                {
                    continue;
                }
                else
                {
                    uchar p2 = V(i-1, j);
                    uchar p3 = V(i-1, j+1);
                    uchar p4 = V(i, j+1);
                    uchar p5 = V(i+1, j+1);

                    uchar p6 = V(i+1, j);
                    uchar p7 = V(i+1, j-1);
                    uchar p8 = V(i, j-1);
                    uchar p9 = V(i-1, j-1);

                    int cn  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                             (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                             (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                             (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);

//                    int cn = (abs(V(i,j+1)-1)-abs((V(i,j+1)-1)*(V(i+1,j)-1)*(V(i+1,j+1)-1)))+
//                            (abs(V(i+1,j)-1)-abs((V(i+1,j)-1)*(V(i+1,j-1)-1)*(V(i,j-1)-1)))+
//                            (abs(V(i,j-1)-1)-abs((V(i,j-1)-1)*(V(i-1,j-1)-1)*(V(i-1,j)-1)))+
//                            (abs(V(i-1,j)-1)-abs((V(i-1,j)-1)*(V(i-1,j+1)-1)*(V(i,j+1)-1)));
                    if (cn==1)
                    {

                        pointToDel.push_back(Point(i,j));
                    }
                    else
                    {
                        continue;
                    }
                }
            }
            else
            {

                continue;
            }
        }
    }


    //mask2
    for (int i = 2; i < fore.rows-2; i++)
    {
        for (int j = 2; j < fore.cols-2; j++)
        {


            if (V(i,j)==1 && V(i,j+1)==1 && V(i,j-1)==0)
            {

                if (V(i+1,j+1)==0 &&
                    V(i+1,j-1)==0 &&
                    V(i+1,j)==0 &&
                    V(i-1,j+1)==0 &&
                    V(i-1,j)==0 &&
                    V(i-1,j-1)==0 )
                {
                    continue;
                }
                else
                {
                    uchar p2 = V(i-1, j);
                    uchar p3 = V(i-1, j+1);
                    uchar p4 = V(i, j+1);
                    uchar p5 = V(i+1, j+1);

                    uchar p6 = V(i+1, j);
                    uchar p7 = V(i+1, j-1);
                    uchar p8 = V(i, j-1);
                    uchar p9 = V(i-1, j-1);

                    int cn  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                             (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                             (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                             (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                    if (cn==1)
                    {

                        pointToDel.push_back(Point(i,j));
                    }
                    else
                    {
                        continue;
                    }
                }
            }
            else
            {

                continue;
            }
        }
    }

    //mask3
    for (int i = 2; i < fore.rows-2; i++)
    {
        for (int j = 2; j < fore.cols-2; j++)
        {


            if (V(i,j)==1 && V(i-1,j)==1 && V(i+1,j)==0)
            {

                if (V(i+1,j+1)==0 &&
                    V(i+1,j-1)==0 &&
                    V(i,j+1)==0 &&
                    V(i,j-1)==0 &&
                    V(i-1,j+1)==0 &&
                    V(i-1,j-1)==0 )
                {
                    continue;
                }
                else
                {
                    uchar p2 = V(i-1, j);
                    uchar p3 = V(i-1, j+1);
                    uchar p4 = V(i, j+1);
                    uchar p5 = V(i+1, j+1);

                    uchar p6 = V(i+1, j);
                    uchar p7 = V(i+1, j-1);
                    uchar p8 = V(i, j-1);
                    uchar p9 = V(i-1, j-1);

                    int cn  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                             (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                             (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                             (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                    if (cn==1)
                    {

                        pointToDel.push_back(Point(i,j));
                    }
                    else
                    {
                        continue;
                    }
                }
            }
            else
            {

                continue;
            }
        }
    }

    //mask4
    for (int i = 2; i < fore.rows-2; i++)
    {
        for (int j = 2; j < fore.cols-2; j++)
        {


            if (V(i,j)==1 && V(i,j-1)==1 && V(i,j+1)==0)
            {

                if (V(i+1,j+1)==0 &&
                        V(i+1,j-1)==0 &&
                        V(i+1,j)==0 &&
                        V(i-1,j+1)==0 &&
                        V(i-1,j)==0 &&
                        V(i-1,j-1)==0 )
                {
                    continue;
                }
                else
                {
                    uchar p2 = V(i-1, j);
                    uchar p3 = V(i-1, j+1);
                    uchar p4 = V(i, j+1);
                    uchar p5 = V(i+1, j+1);

                    uchar p6 = V(i+1, j);
                    uchar p7 = V(i+1, j-1);
                    uchar p8 = V(i, j-1);
                    uchar p9 = V(i-1, j-1);

                    int cn  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                             (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                             (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                             (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                    if (cn==1)
                    {

                        pointToDel.push_back(Point(i,j));
                    }
                    else
                    {
                        continue;
                    }
                }
            }
            else
            {

                continue;
            }
        }
    }

    for (vector<Point>::iterator it = pointToDel.begin(); it!=pointToDel.end();it++)
    {
        V(it->x,it->y)=0;
    }

    if(isFirst) isFirst=false;

 }

    fore *= 255;

}


inline cv::Mat morphSkeleton(cv::Mat& img)
{
    cv::Mat skel(img.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat temp;
    cv::Mat eroded;

    cv::Mat element = cv::getStructuringElement(cv::MORPH_ERODE, cv::Size(3, 3));

    bool done;
    do
    {
      cv::erode(img, eroded, element);
      cv::dilate(eroded, temp, element); // temp = open(img)
      cv::subtract(img, temp, temp);
      cv::bitwise_or(skel, temp, skel);
      eroded.copyTo(img);

      done = (cv::countNonZero(img) == 0);
    } while (!done);

    return skel;



}


