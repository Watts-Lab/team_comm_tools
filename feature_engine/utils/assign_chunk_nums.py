import numpy as np
import pandas as pd

def reduce_chunks(num_rows, max_num_chunks):
    if (num_rows < max_num_chunks * 2):
        max_num_chunks = int(num_rows / 2)
    if max_num_chunks < 1:
        return 1
    else:
        return max_num_chunks
    

# Assign chunk numbers to the chats within each conversation
def create_chunks_messages(chat_data, num_chunks):

    # Calculate the total number of rows per conversation
    conversation_lengths = chat_data.groupby('conversation_num').size()

    chunks = conversation_lengths.apply(lambda x: reduce_chunks(x, num_chunks))

    # Calculate the chunk size based on the total number of conversations
    chunk_size = np.ceil(conversation_lengths / chunks) 
    
    for i, group in chat_data.groupby('conversation_num'): # for each group
        chunk_num = 0
        counter = 0

        for chat_id in group.index.values: # iterate over the index values
            chat_data.at[chat_id, 'chunk_num'] = str(chunk_num)

            counter += 1

            #if counter = 1 for the last row of a group (implies last chunk has one element), and the chunk num > 0, then just make the last one - 1
            if counter == 1 and chunk_num > 0 and chat_id == group.index.values[-1]:
                chat_data.at[chat_id, 'chunk_num'] = str(chunk_num - 1)

            if counter == chunk_size[i] and chunk_num < chunks[i] - 1: # assign any extras to the last chunk
                chunk_num += 1
                counter = 0    

    return(chat_data)


# Assign chunk numbers based on time
def create_chunks(chat_data,num_chunks):

    #check if there are timestamps
    final_df = pd.DataFrame(columns=chat_data.columns)

    for index, conv in chat_data.groupby(['batch_num', 'round_num']):
        
        # Typecheck: str --> convert to DateTime
        isDT = isinstance(type(conv['timestamp'].iloc[0]), str)
        
        if (isDT):
            conv['timestamp'] = pd.to_datetime(conv['timestamp'])

        # Calculate the total duration of the conversation
        total_duration = int((conv['timestamp'].max() - conv['timestamp'].min()).total_seconds()) if isDT else int(conv['timestamp'].max() - conv['timestamp'].min())

        # Calculate the duration of each chunk
        chunk_duration = total_duration / num_chunks

        if chunk_duration == 0:
            chunk_duration = 1

        # Add a new column for chunk number
        conv['chunk'] = -1 

        # Assign the chunk number for each row
        for index, row in conv.iterrows():
            #get the timestamp 
            timestamp = row['timestamp']

            #calculate the chunk number
            chunk_number = int(((timestamp - conv['timestamp'].min())).total_seconds() / chunk_duration) if isDT else int(((timestamp - conv['timestamp'].min())) / chunk_duration)

            #restrict the range of the chunks from 0 to num_chunks - 1
            if chunk_number >= num_chunks:
                conv.at[index, 'chunk'] = num_chunks - 1
            else:
                conv.at[index, 'chunk'] = chunk_number
        final_df = pd.concat([final_df, conv], ignore_index = True)
    
    return final_df


def assign_chunk_nums(chat_data, num_chunks):
    if 'timestamp' in chat_data.columns:
        return create_chunks(chat_data, num_chunks)
    else:
        return create_chunks_messages(chat_data, num_chunks)