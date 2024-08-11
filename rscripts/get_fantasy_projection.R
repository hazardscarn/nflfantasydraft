library(ffanalytics)
library(dplyr)
library(rio)
library(stringi)
library(tidyverse)
library(glue)

setwd("D:/Work/Github/nflfantasydraft/")

my_scrape <- scrape_data(src = c("CBS"), 
                         pos = c("QB","RB","WR","TE","K","DST"),
                         season = 2025,
                         "ppr") # NULL brings in the current week



#Get QB data
qb=my_scrape$QB
rb=my_scrape$RB
wr=my_scrape$WR
te=my_scrape$TE
k=my_scrape$K
dst=my_scrape$DST



colnames(qb)
df <- rb %>%
  mutate(
    ppr_projection = 
      (rush_yds * 0.1) +     # 0.1 points per rushing yard
      (rush_tds * 6) +       # 6 points per rushing touchdown
      (rec * 1) +            # 1 point per reception (PPR)
      (rec_yds * 0.1) +      # 0.1 points per receiving yard
      (rec_tds * 6) -        # 6 points per receiving touchdown
      (fumbles_lost * -2)    # -2 points per fumble lost
  )

cbs_qb=qb%>%filter(data_src=='CBS')%>%
  mutate(
    ppr_projection = (pass_yds * 0.04) +     # 0.04 points per passing yard
      (pass_tds * 6) +       # 6 points per passing touchdown
      (pass_int * -1) +      # -1 point per interception thrown
      (rush_yds * 0.1) +     # 0.1 points per rushing yard
      (rush_tds * 6) +       # 6 points per rushing touchdown
      (fumbles_lost * -2)    # -2 points per fumble lost
  )%>%
  select(player,src_id,id,pos,team,site_pts,ppr_projection)%>%
  rename(cbs_projection=site_pts)

cbs_rb=rb%>%filter(data_src=='CBS')%>%
  mutate(
    ppr_projection = 
      (rush_yds * 0.1) +     # 0.1 points per rushing yard
      (rush_tds * 6) +       # 6 points per rushing touchdown
      (rec * 1) +            # 1 point per reception (PPR)
      (rec_yds * 0.1) +      # 0.1 points per receiving yard
      (rec_tds * 6) -        # 6 points per receiving touchdown
      (fumbles_lost * -2)    # -2 points per fumble lost
  )%>%
  select(player,src_id,id,pos,team,site_pts,ppr_projection)%>%
  rename(cbs_projection=site_pts)

cbs_wr=wr%>%filter(data_src=='CBS')%>%
  mutate(
  ppr_projection = 
    (rush_yds * 0.1) +     # 0.1 points per rushing yard
    (rush_tds * 6) +       # 6 points per rushing touchdown
    (rec * 1) +            # 1 point per reception (PPR)
    (rec_yds * 0.1) +      # 0.1 points per receiving yard
    (rec_tds * 6) -        # 6 points per receiving touchdown
    (fumbles_lost * -2)    # -2 points per fumble lost
)%>%
  select(player,src_id,id,pos,team,site_pts,ppr_projection)%>%
  rename(cbs_projection=site_pts)

cbs_te=te%>%filter(data_src=='CBS')%>%
  mutate(
    ppr_projection = 
      (rec * 1) +            # 1 point per reception (PPR)
      (rec_yds * 0.1) +      # 0.1 points per receiving yard
      (rec_tds * 6) -        # 6 points per receiving touchdown
      (fumbles_lost * -2)    # -2 points per fumble lost
  )%>%
  select(player,src_id,id,pos,team,site_pts,ppr_projection)%>%
  rename(cbs_projection=site_pts)

cbs_k=k%>%filter(data_src=='CBS')%>%
  select(player,src_id,id,pos,team,site_pts)%>%
  rename(cbs_projection=site_pts)%>%
  mutate(ppr_projection=cbs_projection)

cbs_dst=dst%>%filter(data_src=='CBS')%>%
  mutate(player=team,pos='DEF')%>%
  select(player,src_id,id,pos,team,site_pts)%>%
  rename(cbs_projection=site_pts)%>%
  mutate(ppr_projection=cbs_projection)



final=bind_rows(cbs_qb,cbs_wr,cbs_rb,cbs_te,cbs_k,cbs_dst)



##Get ID populated proerly
my_scrape2 <- scrape_data(src = c("CBS", "NFL", "NumberFire"), 
                         pos = c("QB","RB","WR","TE","K","DST"),
                         season = 2025,
                         week = 1) # NULL brings in the current week


#Get QB data
qb=my_scrape2$QB
rb=my_scrape2$RB
wr=my_scrape2$WR
te=my_scrape2$TE
k=my_scrape2$K
dst=my_scrape2$DST

df=bind_rows(qb%>%select(id,player),
             wr%>%select(id,player),
             rb%>%select(id,player),
             te%>%select(id,player),
             k%>%select(id,player))


df2=df%>%group_by(player)%>%
  arrange(desc(id))%>%
  filter(row_number()==1)


final_missing=final%>%filter(is.na(id)==1)
final_missing2=left_join(final_missing%>%select(-id),df2,by=c("player"))
final_df=bind_rows(final%>%filter(is.na(id)==0),final_missing2)


raw_proj=import("data/nfl_projection_raw.csv")
raw_proj=raw_proj%>%filter(avg_type=='robust')


final2=left_join(final,raw_proj%>%select(id,tier,pos_rank,floor_rank,ceiling_rank,rank)%>%
                   mutate(id=as.character(id)),by=c('id'))

export(final2,"data/fantasy_projection_cbs.csv")
