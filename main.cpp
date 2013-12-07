#include<opencv2/highgui/highgui.hpp>
#include<opencv/cvaux.h>

#include<cmath>
#include"gui.hpp"

using namespace std;

const short threshold =85;//80 for test 6  110 for test 3
inline void setForeground(cv::Mat& back, cv::Mat& curr, cv::Mat& fore);
inline void getSkeleton(cv::Mat& fore);
inline void StentifordThinning(cv::Mat& fore);

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

int main(int argc, char* argv[])
{


    QApplication app(argc,argv);



    CvCapture* capture = NULL;

    //capture = cvCaptureFromCAM(0);
    capture = cvCaptureFromFile("test6.webm");


    cv::Mat curr_frame,background,testImg(480,640,CV_8U);
    cv::Mat foreground(480,640,CV_8U);


    //foreground.create(480,640,CV_8U);



    GUI gui;
    gui.show();

    cv::namedWindow("Camera");

    cv::namedWindow("Background");

    //cv::namedWindow("Fore");

    cv::moveWindow("Camera",200,10);
    cv::moveWindow("Background",200,10);



//    cv::BackgroundSubtractorMOG2 bgsub;
//    std::vector<std::vector<cv::Point> > contours;


        while(true)
    {


        curr_frame = cvQueryFrame(capture);

        cv::imshow("Camera",curr_frame);



        if (gui.flagNewBack)
        {
            curr_frame.copyTo(background);

            gui.flagNewBack = false;

            //qDebug()<<background.data[640*480+25];

        }
        if (gui.flag)
        {


              setForeground(background,curr_frame,foreground);
              cv::erode(foreground,foreground,cv::Mat());
//              cv::erode(foreground,foreground,cv::Mat());
//              cv::erode(foreground,foreground,cv::Mat());
//              cv::erode(foreground,foreground,cv::Mat());
//              cv::erode(foreground,foreground,cv::Mat());
//              cv::dilate(foreground,foreground,cv::Mat());
              cv::dilate(foreground,foreground,cv::Mat());
              cv::dilate(foreground,foreground,cv::Mat());
               cv::dilate(foreground,foreground,cv::Mat());
              cv::dilate(foreground,foreground,cv::Mat());
              cv::dilate(foreground,foreground,cv::Mat());
              cv::dilate(foreground,foreground,cv::Mat());

              //for test6 - 1 er 1 dil
              //for test3 - 1 er 7 dil

             //       thinning(foreground);
            //getSkeleton(foreground);

              StentifordThinning(foreground);
              cv::imshow("Stentiford",foreground);

//            cv::imshow("Fore",foreground);
//            cv::imshow("Background",background);

        }

        if(gui.flagTestImg)
        {
            foreground.copyTo(testImg);
            gui.flagTestImg=false;
            break;
        }

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

//        getSkeleton(testImg);
        cv::Mat ZhangSuen,GuoHall,Rosenfeld,Stentiford;

//        testImg.copyTo(ZhangSuen);
//        testImg.copyTo(GuoHall);
//        testImg.copyTo(Rosenfeld);
        testImg.copyTo(Stentiford);
//        StentifordThinning(Stentiford);
//        cv::imshow("Stentiford",Stentiford);

//        thinning(ZhangSuen);
//        thinningGuoHall(GuoHall);
//        getSkeleton(Rosenfeld);
//        cv::imshow("Zhang-Suen",ZhangSuen);
//        cv::imshow("Guo-Hall",GuoHall);
//        cv::imshow("Rosenfeld",Rosenfeld);
cvWaitKey(300000);


    cvReleaseCapture(&capture);
    cvDestroyAllWindows();

    return app.exec();
}

inline void setForeground(cv::Mat& back, cv::Mat& curr, cv::Mat& fore)
{

    //qDebug()<<"fore = "<<fore.channels()<<" curr = "<<curr.channels();
    //cv::Mat cg,bg;
    //cv::cvtColor(curr,cg,CV_BGR2GRAY);
    //cv::cvtColor(back,bg,CV_BGR2GRAY);
    //qDebug()<<"fore = "<<fore.channels()<<" curr = "<<curr.channels();
    for (int i=1;i<back.rows-1;i++)
      {
          for (int j=1;j<back.cols-1;j++)
          {
//              cv::Scalar color_c = curr.at<unsigned char>(i,j);
//              cv::Scalar color_b = back.at<unsigned char>(i,j);

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
    qDebug()<<"begin";
    int high = 1, low = 479, right = 639, left = 1;
    int dir = 0;
    while(dir<600)
    {

        switch(dir%4)
        {
        //сверху вниз
        case 0:

            for (int i = 1; i < fore.cols-1; i++)
            {

                for (int j = 1; j < fore.rows - 1; j++)
                {
                    int num = 0;

                    if (fore.at<unsigned char>(j,i)==0)
                    {
                        continue;
                    }
                    else
                    {
                        if (fore.at<unsigned char>(j,i-1)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(j,i+1)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(j-1,i)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(j+1,i)==255)
                        {
                            num++;
                        }
                        //------
                        if (fore.at<unsigned char>(j+1,i+1)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(j+1,i-1)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(j-1,i+1)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(j-1,i-1)==255)
                        {
                            num++;
                        }


                        if (num > 2 || num==0)
                        {
                            fore.at<unsigned char>(j,i)=0;
                            break;
                        }
                        else
                        {
                            continue;
                        }
//                        if(maxHigh>j)
//                            maxHigh = j;

                    }
                }
            }

            //qDebug()<<"case 0";
//          high=maxHigh;
            high++;

            break;


        //снизу вверх
        case 1:

            for (int i = 1; i < fore.cols-1; i++)
            {

                for (int j = fore.rows-1; j > 1 ; j--)
                {
                    int num = 0;
                    if (fore.at<unsigned char>(j,i)==0)
                    {
                        continue;
                    }
                    else
                    {
                        if (fore.at<unsigned char>(j,i-1)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(j,i+1)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(j-1,i)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(j+1,i)==255)
                        {
                            num++;
                        }
                        //----------
                        if (fore.at<unsigned char>(j+1,i+1)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(j+1,i-1)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(j-1,i+1)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(j-1,i-1)==255)
                        {
                            num++;
                        }

                        if (num > 2 || num==0)
                        {
                            fore.at<unsigned char>(j,i)=0;
                            break;
                        }
                        else
                        {
                            continue;
                        }
                    }
                }
            }
            //low=minLow;
            //qDebug()<<"case 1";
            low--;
            break;

        //слева направо
        case 2:

            for (int i = 1; i < fore.rows-1; i++)
            {

                for (int j = 1; j < fore.cols - 1; j++)
                {
                    int num = 0;
                    if (fore.at<unsigned char>(i,j)==0)
                    {
                        continue;
                    }
                    else
                    {
                        if (fore.at<unsigned char>(i,j-1)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(i,j+1)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(i-1,j)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(i+1,j)==255)
                        {
                            num++;
                        }
                        //----------
                        if (fore.at<unsigned char>(i-1,j-1)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(i-1,j+1)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(i+1,j+1)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(i+1,j-1)==255)
                        {
                            num++;
                        }


                        if (num > 2 || num==0)
                        {
                            fore.at<unsigned char>(i,j)=0;
                            break;
                        }
                        else
                        {
                            continue;
                        }
                    }
                }
            }
            left++;


            break;

            //справа налево
        case 3:

            for (int i = 1; i < fore.rows-1; i++)
            {

                for (int j = fore.cols - 1; j > 1; j--)
                {
                    int num = 0;
                    if (fore.at<unsigned char>(i,j)==0)
                    {
                        continue;
                    }
                    else
                    {
                        if (fore.at<unsigned char>(i,j-1)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(i,j+1)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(i-1,j)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(i+1,j)==255)
                        {
                            num++;
                        }
                        //----------
                        if (fore.at<unsigned char>(i-1,j-1)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(i-1,j+1)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(i+1,j+1)==255)
                        {
                            num++;
                        }
                        if (fore.at<unsigned char>(i+1,j-1)==255)
                        {
                            num++;
                        }

                        if (num > 2 || num==0)
                        {
                            fore.at<unsigned char>(i,j)=0;
                            break;
                        }
                        else
                        {
                            continue;
                        }
                    }
                }
            }
            right--;

            break;

        }
        dir++;
    }

        qDebug()<<"end";

}

#define V fore.at<uchar>

inline void StentifordThinning(cv::Mat& fore)
{
    fore /= 255;
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

                    int cn = (V(i,j+1)-V(i,j+1)*V(i+1,j)*V(i+1,j+1))+
                            (V(i+1,j)-V(i+1,j)*V(i+1,j-1)*V(i,j-1))+
                            (V(i,j-1)-V(i,j-1)*V(i-1,j-1)*V(i-1,j))+
                            (V(i-1,j)-V(i-1,j)*V(i-1,j+1)*V(i,j+1));
                    if (cn==1)
                    {

                        V(i,j)=0;
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

                    int cn = (V(i,j+1)-V(i,j+1)*V(i+1,j)*V(i+1,j+1))+
                            (V(i+1,j)-V(i+1,j)*V(i+1,j-1)*V(i,j-1))+
                            (V(i,j-1)-V(i,j-1)*V(i-1,j-1)*V(i-1,j))+
                            (V(i-1,j)-V(i-1,j)*V(i-1,j+1)*V(i,j+1));
                    if (cn==1)
                    {

                        V(i,j)=0;
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

                    int cn = (V(i,j+1)-V(i,j+1)*V(i+1,j)*V(i+1,j+1))+
                            (V(i+1,j)-V(i+1,j)*V(i+1,j-1)*V(i,j-1))+
                            (V(i,j-1)-V(i,j-1)*V(i-1,j-1)*V(i-1,j))+
                            (V(i-1,j)-V(i-1,j)*V(i-1,j+1)*V(i,j+1));
                    if (cn==1)
                    {

                        V(i,j)=0;
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

                    int cn = (V(i,j+1)-V(i,j+1)*V(i+1,j)*V(i+1,j+1))+
                            (V(i+1,j)-V(i+1,j)*V(i+1,j-1)*V(i,j-1))+
                            (V(i,j-1)-V(i,j-1)*V(i-1,j-1)*V(i-1,j))+
                            (V(i-1,j)-V(i-1,j)*V(i-1,j+1)*V(i,j+1));
                    if (cn==1)
                    {

                        V(i,j)=0;
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

    fore *= 255;



}


