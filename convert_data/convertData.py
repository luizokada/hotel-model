class ConvertData:
    def __init__(self, data):
        self.data = data.drop('Booking_ID', axis=1)
        self.data['type_of_meal_plan'] = self.data.apply(self.meal_plan_convert, axis=1)
        self.data['room_type_reserved'] = self.data.apply(self.room_type_convert, axis=1)
        self.data['market_segment_type'] = self.data.apply(self.market_segment_convert, axis=1)
        self.data['booking_status'] = self.data.apply(self.booking_status_convert, axis=1)

    def meal_plan_convert(self,row):
        if row.type_of_meal_plan == 'Not Selected':
            return 0
        elif row.type_of_meal_plan == 'Meal Plan 1':
            return 1
        elif row.type_of_meal_plan == 'Meal Plan 2':
            return 2
        elif row.type_of_meal_plan == 'Meal Plan 3':
            return 3
        else:
            print("MUST BE MEAL PLAN 1, 2, 3 OR NOT SELECTED")
        
    def room_type_convert(self,row):
        if row.room_type_reserved == 'Room_Type 1':
            return 1
        elif row.room_type_reserved == 'Room_Type 2':
            return 2
        elif row.room_type_reserved == 'Room_Type 3':
            return 3
        elif row.room_type_reserved == 'Room_Type 4':
            return 4
        elif row.room_type_reserved == 'Room_Type 5':
            return 5
        elif row.room_type_reserved == 'Room_Type 6':
            return 6
        elif row.room_type_reserved == 'Room_Type 7':
            return 7
        else:
            print("MUST BE ROOM_TYPE 1, 2, 3, 4, 5, 6 OR 7")
        
    def market_segment_convert(self,row):
        if row.market_segment_type == 'Online':
            return 1
        elif row.market_segment_type == 'Offline':
            return 2
        elif row.market_segment_type == 'Corporate':
            return 3
        elif row.market_segment_type == 'Complementary':
            return 4
        elif row.market_segment_type == 'Aviation':
            return 5
        else:
            print("MUST BE ONLINE, OFFLINE, CORPORATE, COMPLEMENTARY OR AVIATION")
    
    def booking_status_convert(self,row):
        if row.booking_status == 'Not_Canceled':
            return 0
        elif row.booking_status == 'Canceled':
            return 1
        else:
            print("MUST BE CANCELED OR NOT CANCELED")