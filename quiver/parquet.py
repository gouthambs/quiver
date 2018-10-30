import shutil
import os

import pandas as pd
from pyarrow import Table, compat, Array, Column
from pyarrow.compat import guid
from pyarrow.parquet import _get_fs_from_path, _ensure_filesystem, _mkdir_if_not_exists, \
    read_table, write_table


def _to_category_cols(df, categories):
    category_cols = categories or list(df.dtypes.loc[df.dtypes == 'category'].index)
    category_cols = {c: df.loc[:, c].cat.categories for c in category_cols}
    return category_cols


def _upsert_dataframes(df, old_df):
    df = df.loc[~df.index.duplicated(keep='first')]  # remove dupes in subgroup just to be sure
    sub_cols = df.columns
    old_sub_cols = old_df.columns
    if set(sub_cols) != set(old_sub_cols):
        raise ValueError('The columns in old and new groups do not match')
    dft = pd.DataFrame(index=df.index)
    dft['__new__data__'] = 1.0
    result = old_df.join(dft, how='outer')
    upd_rows = pd.notnull(result.__new__data__)
    result.loc[upd_rows, sub_cols] = df
    df = result[sub_cols].copy()
    if len(df.loc[df.index.duplicated()]):
        raise ValueError('Unexpected duplicates found in resulting dataset')
    return df


def upsert_to_dataset(table, root_path, partition_cols=None,
                      filesystem=None, preserve_index=True,
                      temp_folder=None, categories=None, **kwargs):
    if filesystem is None:
        fs = _get_fs_from_path(root_path)
    else:
        fs = _ensure_filesystem(filesystem)

    _mkdir_if_not_exists(fs, root_path)
    if temp_folder:
        if not os.path.exists(temp_folder):
            temp_folder = None

    if partition_cols is not None and len(partition_cols) > 0:
        # df is the data in the new table
        df = table.to_pandas()
        partition_keys = [df[col] for col in partition_cols]
        data_df = df.drop(partition_cols, axis='columns')
        data_cols = df.columns.drop(partition_cols)
        if len(data_cols) == 0:
            raise ValueError("No data left to save outside partition columns")
        subschema = table.schema
        # ARROW-2891: Ensure the output_schema is preserved when writing a
        # partitioned dataset
        for partition_col in partition_cols:
            subschema = subschema.remove(
                subschema.get_field_index(partition_col))
        for keys, subgroup in data_df.groupby(partition_keys):
            if not isinstance(keys, tuple):
                keys = (keys,)
            subdir = "/".join(
                ["{colname}={value}".format(colname=name, value=val)
                 for name, val in zip(partition_cols, keys)])

            prefix = "/".join([root_path, subdir])
            _mkdir_if_not_exists(fs, prefix)
            existing_files = [f for f in os.listdir(prefix) if f.endswith('.parquet')]
            if len(existing_files) > 1:
                raise ValueError('Unsupported scenario, multiple files found in path %s' % prefix)
            if len(existing_files) == 1:
                outfile = existing_files[0]
                full_path = "/".join([prefix, outfile])
                old_table = read_table(full_path)
                category_cols = _to_category_cols(subgroup, categories)  # get categories before merging
                old_subgroup = old_table.to_pandas()
                # TODO: compare old schema with new
                subgroup = _upsert_dataframes(subgroup, old_subgroup)
                # subgroup = pd.concat([subgroup, old_subgroup[~old_subgroup.index.isin(subgroup.index.values)]])
                for c, v in category_cols.items():
                    subgroup.loc[:, c] = subgroup.loc[:, c].astype('category', categories=v)
            else:
                outfile = compat.guid() + ".parquet"
                full_path = "/".join([prefix, outfile])
            subtable = Table.from_pandas(subgroup,
                                         preserve_index=preserve_index,
                                         schema=subschema)
            write_file = os.path.join(temp_folder, outfile) if temp_folder else full_path
            with fs.open(write_file, 'wb') as f:
                write_table(subtable, f, **kwargs)
            if temp_folder:
                shutil.move(write_file, full_path)
    else:
        existing_files = [f for f in os.listdir(root_path) if f.endswith('.parquet')]
        if len(existing_files) > 1:
            raise ValueError('Unsupported scenario, multiple files found in path %s' % root_path)
        if len(existing_files) == 1:
            # append use case
            outfile = existing_files[0]
            full_path = "/".join([root_path, outfile])
            old_table = read_table(full_path)
            subgroup = table.to_pandas()
            category_cols = _to_category_cols(subgroup, categories)
            old_subgroup = old_table.to_pandas()
            # TODO: compare old schema with new
            subgroup = _upsert_dataframes(subgroup, old_subgroup)
            # subgroup = pd.concat([old_subgroup[~old_subgroup.index.isin(subgroup.index)], subgroup])
            for c, v in category_cols.items():
                subgroup.loc[:, c] = subgroup.loc[:, c].astype('category', categories=v)
            schema = table.schema
            table = Table.from_pandas(
                subgroup,
                preserve_index=preserve_index,
                schema=schema
            )
        else:
            # write use case
            outfile = compat.guid() + ".parquet"
            full_path = "/".join([root_path, outfile])

        write_file = os.path.join(temp_folder, outfile) if temp_folder else full_path
        with fs.open(write_file, 'wb') as f:
            write_table(table, f, **kwargs)
        if temp_folder:
            shutil.move(write_file, full_path)



def write_to_dataset(table, root_path, partition_cols=None,
                     filesystem=None, preserve_index=True, **kwargs):
    """
    Wrapper around parquet.write_table for writing a Table to
    Parquet format by partitions.
    For each combination of partition columns and values,
    a subdirectories are created in the following
    manner:
    root_dir/
      group1=value1
        group2=value1
          <uuid>.parquet
        group2=value2
          <uuid>.parquet
      group1=valueN
        group2=value1
          <uuid>.parquet
        group2=valueN
          <uuid>.parquet
    Parameters
    ----------
    table : pyarrow.Table
    root_path : string,
        The root directory of the dataset
    filesystem : FileSystem, default None
        If nothing passed, paths assumed to be found in the local on-disk
        filesystem
    partition_cols : list,
        Column names by which to partition the dataset
        Columns are partitioned in the order they are given
    preserve_index : bool,
        Parameter for instantiating Table; preserve pandas index or not.
    **kwargs : dict, kwargs for write_table function.
    """
    if filesystem is None:
        fs = _get_fs_from_path(root_path)
    else:
        fs = _ensure_filesystem(filesystem)

    _mkdir_if_not_exists(fs, root_path)

    if partition_cols is not None and len(partition_cols) > 0:
        #df = table.to_pandas()
        #partition_keys = [df[col] for col in partition_cols]
        partition_keys = [table.column(col) for col in partition_cols]
        data_table = table.drop(partition_cols )
        #data_cols = df.columns.drop(partition_cols)
        #if len(data_cols) == 0:
        #    raise ValueError('No data left to save outside partition columns')
        subschema = table.schema
        # ARROW-2891: Ensure the output_schema is preserved when writing a
        # partitioned dataset
        for partition_col in partition_cols:
            subschema = subschema.remove(
                subschema.get_field_index(partition_col))
        for keys, subgroup in data_df.groupby(partition_keys):
            if not isinstance(keys, tuple):
                keys = (keys,)
            subdir = '/'.join(
                ['{colname}={value}'.format(colname=name, value=val)
                 for name, val in zip(partition_cols, keys)])
            subtable = pa.Table.from_pandas(subgroup,
                                            preserve_index=preserve_index,
                                            schema=subschema,
                                            safe=False)
            prefix = '/'.join([root_path, subdir])
            _mkdir_if_not_exists(fs, prefix)
            outfile = guid() + '.parquet'
            full_path = '/'.join([prefix, outfile])
            with fs.open(full_path, 'wb') as f:
                write_table(subtable, f, **kwargs)
    else:
        outfile = guid() + '.parquet'
        full_path = '/'.join([root_path, outfile])
        with fs.open(full_path, 'wb') as f:
            write_table(table, f, **kwargs)