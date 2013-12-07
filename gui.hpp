#ifndef GUI_HPP
#define GUI_HPP

#include<QApplication>
#include<QPushButton>
#include<QWidget>
#include<QDebug>

class GUI:public QWidget
{
    Q_OBJECT
public:
    bool flag,flagNewBack,flagTestImg;
    GUI(QWidget* parent = 0);

public slots:
    void slot_background();
    void slot_exit();
    void slot_TestImg();

};

#endif // GUI_HPP
