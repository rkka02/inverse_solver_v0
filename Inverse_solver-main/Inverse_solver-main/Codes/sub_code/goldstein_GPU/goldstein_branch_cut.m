function [step1_unwrapp,step2_unwrapp]=goldstein_branch_cut(residue_map,step1_unwrapp,step2_unwrapp)

%% find the number of residue in each phases

residue_idx=find(residue_map)-1;%keep it as double for safety of rounding
%display_vid_fun_simple(gather(residue_map))
%size(residue_idx)
%residue_value=residue_map(residue_idx);

if ~isempty(residue_idx)

residue_z_id=single(floor((residue_idx)/(size(residue_map,1)*size(residue_map,2))));%compute it as double for safety of rounding also carfull index starts at zero

residue_number = histcounts(residue_z_id,gpuArray(0:(size(residue_map,3)))-0.5);
%plot(1:size(residue_map,3),N);%display the reidue per each phases

%% separe by 3D position reordered version if needed for faster search

precomputed_shortest_number=8;
avrage_per_square=precomputed_shortest_number*2;
square_size = int32(ceil(sqrt((residue_number + 1)/avrage_per_square)));

[case_id,residue_x_id,residue_y_id] =create_sort_value_gpu(int32(residue_idx),int32(square_size),int32(size(residue_map)));%case id is the id the case it is in

ordered_residues=sortrows([int32(residue_idx) int32(residue_map(residue_idx+1)) int32(case_id) int32(residue_x_id) int32(residue_y_id) int32(residue_z_id)],[6,3]);
ordered_residues=ordered_residues.';%for faster acces later on

[lookup_table,lookup_z_start]=get_lookup_gpu(ordered_residues,int32(square_size));%

%error('need to add 1 to lookup_z_start ???');
%warning('need to add 1 to lookup_z_start ???');
%display('S1');
[shortest_list]=get_shortest_sort_gpu(ordered_residues,lookup_table,lookup_z_start,square_size,int32(precomputed_shortest_number),int32(size(residue_map)));
%shortest_list(:,:,30)
%error('stop');

%error('return the distance as a float');
%size(ordered_residues)
%size(shortest_list)

%{
color_residue_map=gather(residue_map);inspect_num=456208;615;color_residue_map(ordered_residues(1,inspect_num)+1)=-3;color_residue_map(ordered_residues(1,shortest_list(1,:,inspect_num)+1)+1)=3;display_vid_fun_simple(gather(color_residue_map));
inspect_num
shortest_list(1,:,inspect_num)
ordered_residues(1,inspect_num)
ordered_residues(1,shortest_list(1,:,inspect_num))
shortest_list(2,:,inspect_num)
%}

%color_residue_map=gather(residue_map);lookup_table_positive=lookup_table(lookup_table>0);color_residue_map(ordered_residues(1,gather(lookup_table_positive(:))+1))=4;implay(gather(color_residue_map));
%if want to see what is what
%color_residue_map=residue_map;color_residue_map(ordered_residues(:,1)+1)=ordered_residues(:,3);figure;imagesc(mod(gather(color_residue_map(:,:,1)),64));colormap lines;figure;plot(1:size(residue_map,3),square_size);%display the number of subdivision

cpu_residues=gather(ordered_residues);
cpu_shortest_list=gather(shortest_list);
cpu_square_size=gather(square_size);
cpu_lookup_table=gather(lookup_table);
cpu_lookup_z_start=gather(lookup_z_start);

%size(cpu_residues)
%size(cpu_shortest_list)
%size(cpu_square_size)
%size(cpu_lookup_table)
%size(cpu_lookup_z_start)

%error('need to find why crashes when increasing the cost of connecting to boundary');
%{
size(cpu_residues)
size(cpu_shortest_list)
size(cpu_square_size)
size(cpu_lookup_table)
size(cpu_lookup_z_start)
int32(size(residue_map))
%}
branches=get_branches_goldstein_v2(cpu_residues,cpu_shortest_list,cpu_square_size,cpu_lookup_table,cpu_lookup_z_start,int32(size(residue_map)));
branches=gpuArray(branches);
step1_unwrapp=int32(step1_unwrapp);
step2_unwrapp=int32(step2_unwrapp);
%branches
%size(branches)
%size(step1_unwrapp)
%size(step2_unwrapp)
%min(branches(6,:))
%min(branches(2,:))
%min(branches(4,:))
%let=branches(2,:)-branches(4,:);max(abs(let(branches(2,:)>0)))
[step1_unwrapp,step2_unwrapp]=raster_branches_gpu(branches,step1_unwrapp,step2_unwrapp);
%[step1_display_unwrapp,step2_display_unwrapp]=raster_branches_display_gpu(branches,step1_unwrapp*0,step2_unwrapp*0);display_vid_fun_simple(single(gather(step1_display_unwrapp(:,:,:))));display_vid_fun_simple(single(gather(step1_unwrapp(:,:,:))));display_vid_fun_simple(gather(single(residue_map)));
%plot(1:size(residue_map,3),square_size);%display the number of subdivision
else
    step1_unwrapp=int32(step1_unwrapp);
    step2_unwrapp=int32(step2_unwrapp);
end

end