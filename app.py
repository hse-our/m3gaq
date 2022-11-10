import streamlit as st    
import pandas as pd 
import backend


def main():
    st.set_page_config(layout="wide")
    # st.config.show.disableWatchdogWarning = True
    
    # '''___________________ SIDEBAR ___________________'''
    st.sidebar.image('app/logo.png')
    st.sidebar.title('Клиентский путь | МКБ')
    max_users = st.sidebar.slider('max_users',min_value=10,max_value=100,step=1,)

    
    # '''___________________  MAIN  ___________________'''
    st.title('Клиентский путь | МКБ')
    st.write('Привет! Это команда MegaQuant. Специально для МКБ мы разработали сервис для работы с клиентским путем:')
    

    
    # '''___________________  FILE_UPLOADER  ___________________'''
    file = st.file_uploader('Загрузите csv файл с данными о клиентском путе',type=['csv'])
    if file is not None: 
        st.write()
        df = backend.get_df(pd.read_csv(file))
        
        # process_flag_num = st.sidebar.selectbox(label='process_flag_num',options=['complete'])
        start_product = st.sidebar.multiselect(label='product',options=df.start_product.unique(),default=df.start_product.iloc[0])     
        # day_time = st.sidebar.selectbox(label='day_time',options=df.day_time.unique())     
        # day_of_week = st.sidebar.selectbox(label='day_of_week',options=df.day_of_week.unique())

        df_ = df.copy()
        # df_ = df_[df_.process_flag_num == process_flag_num]
        df_ = df_[df_.start_product.isin(start_product)]

        
        metrics_by_user = backend.get_metrics_by_user(df_)
        sankey_fig = backend.super_function(metrics_by_user, max_users)

        st.plotly_chart(sankey_fig, use_container_width=True)

        url_name = st.sidebar.selectbox(label='url_name',options=df.url_start.unique())
        # popular_paths = backend.find_popular_paths(backend.data_prep(df),url_name)
        # st.write(popular_paths)

        sank = backend.draw_sankey_popular(df,url_name)
        st.write('Топовые пути')

        st.plotly_chart(sank, use_container_width=True)


        st.write('Метрики по страницам')
        
        st.write(metrics_by_user)

        url_name = st.sidebar.selectbox(label='popular_paths',options=['complete'])
    else:
        st.info(
            f"""
                👆 Попробуйте загрузить [data.csv](https://petyaeva.ru/share/data.csv)
                """
        )

    
if __name__ == '__main__':
    main()