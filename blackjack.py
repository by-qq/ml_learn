import random

class Blackjack:
    cards = ["2","3","4","5","6","7","8","9","J","Q","K","A"] * 4
    b_nums = []
    p_nums = []

    def __init__(self,p_input):
        # 庄家的身份是电脑说明并不应该初始化这个identity，所以下面的choose需要分开操作
        self.p_input = p_input      # 玩家输入的内容
        for i in range(2):
            card = random.choice(self.cards)
            card_num = self.r_num(card,sum(self.b_nums))
            self.cards.remove(card)
            self.b_nums.append(card_num)
        for i in range(2):
            card = random.choice(self.cards)
            card_num = self.r_num(card, sum(self.p_nums))
            self.cards.remove(card)
            self.p_nums.append(card_num)

    def r_num(self,card,nums_total):
        # 定义抽取到的不同卡牌的返回内容
        if card.isdigit():
            return int(card)
        elif card in ["J","Q","K"]:
            return 10
        else:   # 抽取到A，需要判定下返回1,还是返回11
            if 21 - nums_total > 11:
                return 11
            else:
                return 1

    def Bust(self,nums:list): # 判定是不是爆点
        if sum(nums) > 21:
            return f"You bust! You lose!"
        else:
            return f"continue"


    # 定义抽取动作：针对不同的玩家有不同的抽取动作
    def banker_choose(self):   # 针对庄家的抽取动作
        # 获取到现在庄家的点数b_num,这里需要有一个字典b_nums
        while True:
            b_nums_total = sum(self.b_nums)
            if b_nums_total < 17:
                # 抽取，并将抽取到的内容放到本身的字典中
                card = random.choice(Blackjack.cards)
                card_num = self.r_num(card,b_nums_total)
                self.cards.remove(card)
                self.b_nums.append(card_num)
                result = self.Bust(self.b_nums)
                if result == "You bust! You lose!":
                    return "The banker lost"
            break


    def player_choose(self,p_input):   # 针对玩家的抽取动作
        p_nums_total = sum(self.p_nums)
        # 需要通过玩家的输入input_content 来判断是否进行抽取
        if p_input == "y":
            # 继续抽取,抽取完成之后需要判断是否爆点（Bust）了，这里需要p_nums
            card = random.choice(Blackjack.cards)
            card_num = self.r_num(card,p_nums_total)
            self.cards.remove(card)
            self.p_nums.append(card_num)
        result = self.Bust(self.p_nums) # 判断是不是爆牌，返回bool
        # 输入内容是n，表示不继续抽取，并且说明没有爆点，可以直接进行比较

        return result

    def b_p_compare(self):  # 如果玩家选择不继续抽取，需要判定玩家和庄家点数谁大
        b_nums_total = sum(self.b_nums)
        p_nums_total = sum(self.p_nums)
        print(f"\nTotal points of the player are：{p_nums_total},The dealer's total points are：{b_nums_total}")
        if b_nums_total > p_nums_total:
            return f"Your hand is weak, you lost."
        elif b_nums_total == p_nums_total:
            return f"You drew with your opponent"
        else:
            return f"You win"


def play_blackjack():
    game = Blackjack(input("Are you ready to start the game?: y / n "))
    while True:
        if game.p_input == "n":
            return
        print(f"Your cards: {game.p_nums})")
        print(f"Dealer's upcard: [{game.b_nums[:-1]},?]")
        # 玩家回合
        while True:
            choice = input("Do you want to hit again?(y/n): ").lower()
            player_result = game.player_choose(choice)
            print(f"Your cards: {game.p_nums} ")

            if player_result in ["You bust! You lose!"]:
                print(f"{player_result}，Dealer's upcard：{game.b_nums}")
                return
            elif player_result == "continue":
                break

        # 庄家回合
        print("Banker's Turn:")
        banker_result = game.banker_choose()
        if banker_result == "The banker lost":
            print(f"{banker_result},Dealer's upcard: {game.b_nums}")
            return
        if choice == "n":
            print(f"Dealer's upcard: {game.b_nums}")
            # 比较点数
            print(game.b_p_compare())
            return

if __name__ == "__main__":

    print("Do you want to play Blackjack? Enter y or n")
    if input() == "y":
        play_blackjack()
    else:
        exit()


