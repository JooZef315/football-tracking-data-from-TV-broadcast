class WorldCupSoccerField
{
  float field_width;
  float field_height;
  float big_circle_radius;
  float goal_length;
  float goal_area_width ;
  float goal_area_height;
  float penalty_area_width;
  float penalty_area_height;
  float corner_circle_radius;
  float penalty_point_offset;
  float draw_scale;
  
  WorldCupSoccerField()
  {
    draw_scale = 10;
    field_width = 105 * draw_scale;
    field_height = 68 * draw_scale;
    big_circle_radius = 9.15 * draw_scale;
    goal_length = 7.32 * draw_scale;
    goal_area_width = goal_length + 2 * 5.5 * draw_scale;
    goal_area_height = 5.5 * draw_scale;
    penalty_area_width = goal_length + 2 * 16.5 * draw_scale;
    penalty_area_height = 16.5 * draw_scale;
    corner_circle_radius = 1 * draw_scale;
    penalty_point_offset = 11 * draw_scale;
  }
  
  void draw()
  {
    noFill();
    stroke(255);
    strokeCap(SQUARE);
    strokeJoin(MITER);
    strokeWeight(7.0);
    draw_touch_line();
    draw_halfway_line();
    draw_goal_area();
    draw_penalty_area();
    draw_center_circle();
    draw_archs();
    draw_corners();
  }
  
  void draw_corners()
  {
    arc(0, 
        0, 
        corner_circle_radius*2, 
        corner_circle_radius*2,
        0,
        HALF_PI);
    arc(field_width, 
        0, 
        corner_circle_radius*2, 
        corner_circle_radius*2,
        HALF_PI,
        PI);
    arc(field_width, 
        field_height,
        corner_circle_radius*2, 
        corner_circle_radius*2,
        PI,
        PI+HALF_PI);
    arc(0, 
        field_height,
        corner_circle_radius*2, 
        corner_circle_radius*2,
        PI+HALF_PI,
        TWO_PI);
  }
  
  void draw_archs()
  {
    float start_angle = cos((penalty_area_height-penalty_point_offset)/big_circle_radius)+radians(5);
    float end_angle = -start_angle;
    arc(penalty_point_offset, 
        field_height/2, 
        big_circle_radius*2, 
        big_circle_radius*2,
        end_angle,
        start_angle);
    arc(field_width-penalty_point_offset, 
        field_height/2, 
        big_circle_radius*2, 
        big_circle_radius*2,
        PI-start_angle,
        PI-end_angle);
  }
  
  void draw_halfway_line()
  {
    line(field_width/2, 0, field_width/2, field_height);
  }
  
  void draw_penalty_area()
  {
    float x1 = 0;
    float x2 = penalty_area_height;
    float y1 = (field_height/2) - (penalty_area_width/2);
    float y2 = (field_height/2) + (penalty_area_width/2);
    beginShape();
    vertex(field_width-x1, y1);
    vertex(field_width-x2, y1);
    vertex(field_width-x2, y2);
    vertex(field_width-x1, y2);
    vertex(field_width-x1, y2);
    endShape();
    beginShape();
    vertex(x1, y1);
    vertex(x2, y1);
    vertex(x2, y2);
    vertex(x1, y2);
    vertex(x1, y2);
    endShape();
  }
  
  void draw_goal_area()
  {
    float x1 = 0;
    float x2 = goal_area_height;
    float y1 = (field_height/2) - (goal_area_width/2);
    float y2 = (field_height/2) + (goal_area_width/2);
    beginShape();
    vertex(field_width-x1, y1);
    vertex(field_width-x2, y1);
    vertex(field_width-x2, y2);
    vertex(field_width-x1, y2);
    vertex(field_width-x1, y2);
    endShape();
    beginShape();
    vertex(x1, y1);
    vertex(x2, y1);
    vertex(x2, y2);
    vertex(x1, y2);
    vertex(x1, y2);
    endShape();
  }
  
  void draw_center_circle()
  {
    circle(field_width/2, field_height/2, big_circle_radius*2);
  }
  
  void draw_touch_line()
  {
    strokeWeight(14.0);
    beginShape();
    vertex(0, 0);
    vertex(field_width, 0);
    vertex(field_width, field_height);
    vertex(0, field_height);
    vertex(0, 0);
    endShape();
    strokeWeight(7.0);
  }
}
