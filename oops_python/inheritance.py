class Car:
    def __init__(self , brand , model) :
        self.brand = brand
        self.model = model
    
class ElectricCar(Car) :
    def __init__(self , brand , model , battery) :
        super().__init__(brand , model)
        self.battery = battery

myTesla = ElectricCar("Tata" , "Punch" , "18Kwh")
print(myTesla.brand)
print(myTesla.model)
print(myTesla.battery)