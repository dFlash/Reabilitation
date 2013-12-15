#include "gui.hpp"

GUI::GUI(QWidget *parent):QWidget(parent)
{
        flag = false;
        flagNewBack = false;
        flagTestImg = false;
        setFixedSize(200,250);
        move(10,10);

        QPushButton* push_background = new QPushButton("Set background",this);
        push_background->setGeometry(10,10,130,30);

        connect(push_background,SIGNAL(clicked()),this,SLOT(slot_background()));

        QPushButton* push_exit = new QPushButton("Exit",this);
        push_exit->setGeometry(10,40,50,30);

        connect(push_exit,SIGNAL(clicked()),this,SLOT(slot_exit()));

        QPushButton* pushTestImg = new QPushButton("TestImg",this);
        pushTestImg->setGeometry(10,70,100,30);

        connect(pushTestImg,SIGNAL(clicked()),this,SLOT(slot_TestImg()));
    }


void GUI::slot_background()
{
    flag = true;
    flagNewBack = true;
}

void GUI::slot_exit()
{
    exit(0);
}

void GUI::slot_TestImg()
{
    flagTestImg = true;
}
