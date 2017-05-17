BEGIN TRANSACTION;
CREATE TABLE "skills" (
	`name`	TEXT,
	`mp`	INTEGER,
	`dmg`	INTEGER,
	`summon`	TEXT,
	`class`	TEXT
);
INSERT INTO `skills` (name,mp,dmg,summon,class) VALUES ('fire',12,16,'No','Black Mage'),
 ('ice',12,16,'No','Black Mage'),
 ('holy',45,65,'No','White Mage'),
 ('cyclops',65,88,'Yes','Summoner'),
 ('haste',10,10,'No','Time Mage'),
 ('wave fist',0,16,'No','Monk'),
 ('weapon break',0,0,'No','Knight');
CREATE TABLE "classes" (
	`class`	TEXT,
	`hp`	INTEGER,
	`mp`	INTEGER,
	`id`	INTEGER
);
INSERT INTO `classes` (class,hp,mp,id) VALUES ('Monk',220,40,1),
 ('Ninja',120,35,2),
 ('Black Mage',75,95,3),
 ('Knight',250,30,4),
 ('White Mage',85,80,5),
 ('Time Mage',80,85,6);
COMMIT;
