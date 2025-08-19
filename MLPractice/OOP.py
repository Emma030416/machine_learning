# # 定义类
# class Car(): # 类的命名遵循驼峰命名法，如 HelloWorld
#     def run(self): # 定义类的方法时必须包含self（谁调用，self就代表谁）
#         print("run")
#     def work(self):
#         self.run() # 类内部，通过self.的方式调用其他成员
#         print("work")
#
# # 测试代码写main里，只在运行该文件时执行，其他文件导入模块时不执行
# if __name__ == '__main__': # 输入main回车就会自动补全
#     # 创建对象
#     car = Car()
#     # 调用方法
#     car.work()
#     # 类外添加的属性只属于当前对象，其他同类对象不具备
#     car.color = "red"
#     print(f"汽车颜色:{car.color}")

# # init无参数
# class Car():
#     def __init__(self):  # 类内用init魔法方法初始化属性
#         self.color = 'red'
#         self.number = 4
#
# if __name__ == '__main__':
#     car1 = Car()
#     print(f"汽车颜色:{car1.color},汽车轮胎数:{car1.number}")
#     car2 = Car()
#     car2.color = 'black'# 修改属性值
#     print(f"汽车颜色:{car2.color},汽车轮胎数:{car2.number}")

# # init有参数
# class Car():
#     def __init__(self, color, number):  # 参数名就和属性名相同，便于理解
#         self.color = color
#         self.number = number
#
# if __name__ == '__main__':
#     car = Car('red', 4) # 在类外创建对象时传入参数
#     print(f"汽车颜色:{car.color},汽车轮胎数:{car.number}")

# # 魔法方法
# class Car():
#     def __init__(self, color, number):  # 参数名就和属性名相同，便于理解
#         self.color = color
#         self.number = number
#     def __str__(self):
#         return f"汽车颜色:{self.color},汽车轮胎数:{self.number}"
#     def __del__(self):
#         print(f"{self}对象被释放了")
#
# if __name__ == '__main__':
#     car = Car('red', 4) # 在类外创建对象时传入参数
#     print(car) # 将car作为self参数传给__str__()方法

# # 减肥案例
# class Student():
#     def __init__(self):
#         self.current_weight = 100
#
#     def run(self):
#         self.current_weight -= 0.5
#         print(f"当前体重为{self.current_weight}kg")
#
#     def eat(self):
#         self.current_weight += 2
#         print(f"当前体重为{self.current_weight}kg")
#
#     def __del__(self):
#         print(f"当前对象{self}已释放")
#
# if __name__ == '__main__':
#     student = Student()
#     student.run()
#     student.eat()

# # 烤地瓜案例
# class SweetPotato():
#     def __init__(self):
#         self.cook_time = 0 # 烘烤的时间，初始值为0
#         self.cook_state = '生的' # 地瓜的状态，初始值为'生的'
#         self.condiments = [] # 添加的调料，初始值为空列表
#
#     def cook(self, time):
#         if time < 0:
#             print('非法时间，请重新传入')
#         else:
#             self.cook_time += time
#
#             if self.cook_time < 3:
#                 self.cook_state = '生的'
#             elif self.cook_time < 7:
#                 self.cook_state = '半生不熟'
#             elif self.cook_time <= 12:
#                 self.cook_state = '熟了'
#             else:
#                 self.cook_state = '已烤焦，糊了'
#
#     def add_condiment(self,condiment):
#         self.condiments.append(condiment)
#
#     def __str__(self):
#         return f'烘烤时间:{self.cook_time}, 地瓜状态:{self.cook_state}, 添加的调料:{self.condiments}'
#
# if __name__ == '__main__':
#     digua = SweetPotato()
#
#     digua.cook(2)
#     digua.cook(5)
#
#     digua.add_condiment('孜然粉')
#     digua.add_condiment('酱汁')
#
#     print(digua)

# # 摊煎饼案例（单继承 + 重写）
# # 师傅类
# class Master(object):
#     def __init__(self):
#         self.kongfu = '古法摊煎饼果子技术'
#     def make_cake(self):
#         print(f'采用{self.kongfu}制作煎饼果子')
# # 徒弟类
# class Prentice(Master):
#     # 重写属性和方法
#     def __init__(self):
#         self.kongfu = '独创煎饼果子技术'
#     # # 调用徒弟自己的方法
#     # def make_cake(self):
#     #     print(f'采用{self.kongfu}制作煎饼果子')
#     # 调用老师傅的方法
#     def make_master_cake(self):
#         super().__init__() # 一定要先初始化父类的属性！！！
#         super().make_cake()
#
# if __name__ == '__main__':
#     p = Prentice()
#     p.make_cake()
#     p.make_master_cake()

# # 摊煎饼案例（多继承 + 重写）
# # 师傅类
# class Master(object):
#     def __init__(self):
#         self.kongfu = '古法摊煎饼果子技术'
#     def make_cake(self):
#         print(f'采用{self.kongfu}制作煎饼果子')
# # 学校类
# class School(object):
#     def __init__(self):
#         self.kongfu = '黑马AI摊煎饼果子技术'
#     def make_cake(self):
#         print(f'采用{self.kongfu}制作煎饼果子')
# # 徒弟类
# class Prentice(School, Master):
#     # 重写属性和方法
#     def __init__(self):
#         self.kongfu = '独创煎饼果子技术'
#     # 调用徒弟自己的方法
#     def make_cake(self):
#         print(f'采用{self.kongfu}制作煎饼果子')
#     # 调用老师傅的方法
#     def make_master_cake(self):
#         Master.__init__(self) # 一定要先初始化父类的属性！！！
#         Master.make_cake(self)
#     # 调用学校的方法
#     def make_school_cake(self):
#         School.__init__(self)
#         School.make_cake(self)
#
# if __name__ == '__main__':
#     p = Prentice()
#     p.make_cake()
#     p.make_master_cake()
#     p.make_school_cake()

# 摊煎饼案例（多层继承）
# # 师傅类
# class Master(object):
#     def __init__(self):
#         self.kongfu = '古法摊煎饼果子技术'
#     def make_cake(self):
#         print(f'采用{self.kongfu}制作煎饼果子')
# # 学校类
# class School(object):
#     def __init__(self):
#         self.kongfu = '黑马AI摊煎饼果子技术'
#     def make_cake(self):
#         print(f'采用{self.kongfu}制作煎饼果子')
# # 徒弟类
# class Prentice(School, Master):
#     # 重写属性和方法
#     def __init__(self):
#         self.kongfu = '独创煎饼果子技术'
#     # 调用老师傅的方法
#     def make_master_cake(self):
#         Master.__init__(self) # 一定要先初始化父类的属性！！！
#         Master.make_cake(self)
#     # 调用学校的方法
#     def make_school_cake(self):
#         School.__init__(self)
#         School.make_cake(self)
# # 徒孙类
# class TuSun(Prentice):
#     pass
#
# if __name__ == '__main__':
#     ts = TuSun()
#     ts.make_cake()
#     ts.make_master_cake()
#     ts.make_school_cake()

# # 私有化
# class Prentice(object):
#     def __init__(self):
#         self.kongfu = '独创煎饼果子技术'
#         self.__money = 500 #私有化money
#     def make_cake(self):
#         print(f'采用{self.kongfu}制作煎饼果子')
#     # 提供公共接口访问money
#     def get_money(self):
#         return self.__money
#     # 提供公共接口修改money
#     def set_money(self, money):
#         self.__money = money
# class TuSun(Prentice):
#     pass
#
# if __name__ == '__main__':
#     ts = TuSun()
#     print(ts.get_money())
#     ts.set_money(600)
#     print(ts.get_money())

# # 多态案例
# # 有继承
# class Animal(object):
#     def speak(self):
#         pass
# class Dog(Animal):
#     def speak(self):
#         print('汪汪汪')
# class Cat(Animal):
#     def speak(self):
#         print('喵喵喵')
# # 有多态
# def make_noise(an : Animal): # 意思是an必须是Animal的对象，或者其子类对象
#     an.speak()
#
# if __name__ == '__main__':
#     # 创建类对象
#     d = Dog()
#     c = Cat()
#     make_noise(d)
#     make_noise(c)

# # 实例方法
# class Student(object):
#     # 类属性
#     teacher_name = '王老师'
#     # 实例方法
#     def method00(self,stu_name):
#         print(f'{self.teacher_name}的学生是{stu_name}') # 通过self访问
#
# if __name__ == '__main__':
#     stu = Student()
#     stu.method00('张三') # 推荐对象名
#     Student.method00(stu,'张三') # 也可以
# #类方法
# class Student(object):
#     # 类属性
#     teacher_name = '王老师'
#     # 类方法
#     @classmethod
#     def method01(cls, stu_name):
#         print(f'{cls.teacher_name}的学生是{stu_name}') # 通过cls访问
#
# if __name__ == '__main__':
#     stu = Student()
#     Student.method01('张三') # 推荐类名
#     stu.method01('张三') # 也可以
# 静态访问
class Student(object):
    # 类属性
    teacher_name = '王老师'
    # 静态方法
    @staticmethod
    def method02(stu_name):
        print(f'{Student.teacher_name}的学生是{stu_name}') # 通过类名访问

if __name__ == '__main__':
    stu = Student()
    Student.method02('张三') # 推荐类名
    stu.method02('张三') # 也可以
















