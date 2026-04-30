class Car :
    def __init__ (self , brand , model) :
        self.__brand = brand 
        self.__model = model


    def get_car(self):
        return self.__brand + " !"
    
    def get_car_all_details(self):
        return self.__brand + " " + self.__model 
    
class ElectricCar(Car) :
    def __init__(self , brand , model , battery) :
        super().__init__(brand , model)
        self.battery = battery

my_car = ElectricCar("Tata" , "Punch" , "18Kwh")
print(my_car.get_car())
print(my_car.get_car_all_details())

# print(my_car.brand)
# print(my_car.model)
# print(my_car.battery)
# print(my_car.get_car())