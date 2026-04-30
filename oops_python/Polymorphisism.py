class Ac :
    def __init__(self , founder , risk) :
        self.founder = founder 
        self.risk = risk

    def details(self):
        return self.founder + " " + self.risk

class Dc(Ac) :
    def __init__(self , founder , risk , distance) :
        super().__init__(founder , risk)
        self.distance = distance

    def details(self):
        return self.founder + " " + self.risk
    
ac = Ac("Nikola Tesla" , "High")
print(ac.details())

dc = Dc("Thomas elva Eddision" , "Low" , "Low")
print(dc.details())