class Car :
    def __init__(self , brand , model) :
        self.brand = brand
        self.model = model

    # def get_brand(self) :
    #     return self.__brand + " !"

    def full_name(self) :
        return f"{self.brand} , {self.model}"

class ElectricCar(Car) :
    def __init__(self , brand , model , battery) :

        super().__init__(brand , model)
        self.battery = battery

my_Tesla = ElectricCar("Tesla" , "Model S" , "18Kwh")

#print(my_Tesla.__brand)
print(my_Tesla.brand)
print(my_Tesla.model)


# print(my_Tesla.brand)
# print(my_Tesla.model)
# print(my_Tesla.battery)

# print(my_Tesla.full_name())

# my_car = Car("Toyota" , "Carola")
# print(my_car.brand)
# print(my_car.model)

# print(my_car.full_name())

# my_new_car = Car("Tata" , "Punch")
# print(my_new_car.brand)
# print(my_new_car.model)\

# print(my_new_car.full_name())