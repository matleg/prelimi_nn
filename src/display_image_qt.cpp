#include <QApplication>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QGraphicsScene scene;
    QGraphicsView view(&scene);
    QGraphicsPixmapItem item(QPixmap("../data/penguin.png"));
    scene.addItem(&item);
    view.show();
    return a.exec();
}