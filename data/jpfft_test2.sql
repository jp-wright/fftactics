BEGIN TRANSACTION;
CREATE TABLE "skills" (
	`name`	TEXT,
	`mp`	INTEGER,
	`dmg`	INTEGER,
	`summon`	TEXT,
	`class`	TEXT
);
INSERT INTO `skills` VALUES ('fire',12,16,'No','Black Mage');
INSERT INTO `skills` VALUES ('ice',12,16,'No','Black Mage');
INSERT INTO `skills` VALUES ('holy',45,65,'No','White Mage');
INSERT INTO `skills` VALUES ('cyclops',65,88,'Yes','Summoner');
INSERT INTO `skills` VALUES ('haste',10,10,'No','Time Mage');
INSERT INTO `skills` VALUES ('wave fist',0,16,'No','Monk');
INSERT INTO `skills` VALUES ('weapon break',0,0,'No','Knight');
CREATE TABLE "classes" (
	`class`	TEXT,
	`hp`	INTEGER,
	`mp`	INTEGER,
	`id`	INTEGER
);
INSERT INTO `classes` VALUES ('Monk',220,40,1);
INSERT INTO `classes` VALUES ('Ninja',120,35,2);
INSERT INTO `classes` VALUES ('Black Mage',75,95,3);
INSERT INTO `classes` VALUES ('Knight',250,30,4);
INSERT INTO `classes` VALUES ('White Mage',85,80,5);
INSERT INTO `classes` VALUES ('Time Mage',80,85,6);
COMMIT;
