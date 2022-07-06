
WorldCupSoccerField soccer_field;

void setup()
{
  size(1050, 680);
  soccer_field = new WorldCupSoccerField();
  noLoop();
  noSmooth();
}

void draw()
{
  background(77,136,53);
  //background(0);
  soccer_field.draw();
  save("pitch_template.png");
}
